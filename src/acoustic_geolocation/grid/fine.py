"""Precision Doppler modeling and scoring: upscale surviving coarse
candidates to a fine grid, synthesize each candidate's modeled Doppler
curve via the numerical propagation solver (not the coarse stage's
closed-form CPA, which only finds a single closest-approach instant),
and score it against the audio stage's own extracted ridge frequencies.

Scoring compares modeled frequency to `DopplerEvent.curve_freqs` (the
ridge's own parabolic-sub-bin-interpolated frequencies) rather than
resampling raw spectrogram brightness: the ridge is already more
frequency-precise than the STFT's bin width, and comparing in Hz avoids
needing this module to re-read the original audio.

The recording's absolute start time T0 is an unknown nuisance parameter,
not inherited from the coarse stage: each fine point re-seeds its own T0
from its own delayed-CPA timing (not the coarse parent's), then refines
it continuously via scalar minimization. The coarse stage's
`t0_hypothesis` is only accurate to the coarse cell's own position
error, which can be tens of meters to kilometers off the fine point
actually being scored.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ..physics.constants import SPEED_OF_SOUND_MPS
from ..physics.doppler import doppler_shift, radial_velocity
from ..physics.geodesy import DemSource, geodetic_to_ecef
from ..physics.propagation import solve_emission_time
from ..types import Candidate, DopplerEvent, FlightTrack, GridPoint, ScoredCandidate
from .coarse import delayed_cpa

_METERS_PER_DEGREE_LAT = 111_320.0


def upscale_grid(
    candidates: list[Candidate], coarse_resolution_m: float, fine_resolution_m: float, dem: DemSource
) -> list[Candidate]:
    """Replace each surviving coarse candidate with a cluster of finer
    points tiling its own `[-coarse_resolution/2, +coarse_resolution/2]`
    cell -- not a radius neighbourhood -- so adjacent coarse survivors'
    fine points don't overlap, and the true location is guaranteed to
    land inside its nearest coarse survivor's own cell.
    """
    half_width_m = coarse_resolution_m / 2.0
    fine_candidates: list[Candidate] = []

    def _cell_centered_offsets(half_width_deg: float, step_deg: float) -> np.ndarray:
        # Cell-centered, not edge-to-edge-inclusive: an edge-inclusive
        # np.arange(-half, half, step) puts points exactly on a cell's
        # boundary, so two adjacent coarse survivors' fine tilings would
        # duplicate the shared edge. (np.arange(n) + 0.5) * step - half
        # instead centers each fine cell within the coarse cell.
        n = max(int(round(2.0 * half_width_deg / step_deg)), 1)
        return (np.arange(n) + 0.5) * step_deg - half_width_deg

    for coarse in candidates:
        meters_per_degree_lon = _METERS_PER_DEGREE_LAT * max(np.cos(np.radians(coarse.point.lat)), 1e-6)
        lat_step = fine_resolution_m / _METERS_PER_DEGREE_LAT
        lon_step = fine_resolution_m / meters_per_degree_lon
        lat_half_deg = half_width_m / _METERS_PER_DEGREE_LAT
        lon_half_deg = half_width_m / meters_per_degree_lon

        lat_offsets = _cell_centered_offsets(lat_half_deg, lat_step)
        lon_offsets = _cell_centered_offsets(lon_half_deg, lon_step)

        for dlat in lat_offsets:
            for dlon in lon_offsets:
                lat, lon = coarse.point.lat + dlat, coarse.point.lon + dlon
                elevation_m = float(dem.elevation_m(lat, lon))
                point = GridPoint(
                    lat=float(lat), lon=float(lon), elevation_m=elevation_m,
                    ecef=geodetic_to_ecef(lat, lon, elevation_m),
                )
                fine_candidates.append(
                    Candidate(
                        point=point,
                        flight_1=coarse.flight_1,
                        flight_2=coarse.flight_2,
                        t0_hypothesis=coarse.t0_hypothesis,
                        coarse_time_offset_error=coarse.coarse_time_offset_error,
                    )
                )

    return fine_candidates


def model_doppler_curve(
    point: GridPoint, flight: FlightTrack, f0: float, t_heard: np.ndarray, c: float = SPEED_OF_SOUND_MPS
) -> np.ndarray:
    """Synthesize the theoretical Doppler curve `flight` would produce at
    `point`, sampled at the given (absolute) heard times."""
    t_emission = solve_emission_time(t_heard, point.ecef, flight.position_ecef, flight.velocity_ecef, c=c)
    position = flight.position_ecef(t_emission)
    velocity = flight.velocity_ecef(t_emission)
    v_r = radial_velocity(velocity, position, point.ecef)
    return doppler_shift(f0, v_r, c)


def _seed_t0(
    point: GridPoint, flight_1: FlightTrack, flight_2: FlightTrack,
    event_1: DopplerEvent, event_2: DopplerEvent, c: float,
) -> float:
    t_heard_1, _ = delayed_cpa(point, flight_1, c)
    t_heard_2, _ = delayed_cpa(point, flight_2, c)
    return 0.5 * ((t_heard_1 - event_1.t_heard_ref) + (t_heard_2 - event_2.t_heard_ref))


_MIN_VALID_FRACTION = 0.9


def _residual_cost(
    t0: float, point: GridPoint, flight_1: FlightTrack, flight_2: FlightTrack,
    event_1: DopplerEvent, event_2: DopplerEvent, c: float,
) -> float:
    cost = 0.0
    for flight, event in ((flight_1, event_1), (flight_2, event_2)):
        modeled = model_doppler_curve(point, flight, event.f0_estimate, event.curve_times + t0, c)
        valid = ~np.isnan(modeled)
        # Require most of the ridge to be explainable by this candidate,
        # not just whatever few samples converged -- averaging only a
        # small valid remainder would let a mostly-NaN (e.g. mostly
        # out-of-track-range) candidate look artificially good.
        if np.mean(valid) < _MIN_VALID_FRACTION:
            return float(np.inf)
        cost += float(np.mean((modeled[valid] - event.curve_freqs[valid]) ** 2))
    return cost


def _residual_cost_delta(
    delta_t0: float, t0_seed: float, point: GridPoint, flight_1: FlightTrack, flight_2: FlightTrack,
    event_1: DopplerEvent, event_2: DopplerEvent, c: float,
) -> float:
    """Same cost, parametrized as a small offset from `t0_seed` rather
    than an absolute epoch-scale T0. scipy's bounded ("fminbound") Brent
    minimizer sizes its convergence tolerance relative to the magnitude
    of its argument (`sqrt(eps) * abs(x) + xatol/3`); at epoch-scale
    (~1.7e9), `sqrt(eps) * abs(x)` alone is on the order of tens of
    seconds, so the minimizer accepts its very first golden-section probe
    as converged, nowhere near the true minimum. Optimizing over a small
    delta keeps the argument near zero, where that tolerance is actually
    tight. Any scalar optimizer over an absolute epoch-scale time needs
    the same offset-from-seed treatment -- see CLAUDE.md.
    """
    return _residual_cost(t0_seed + delta_t0, point, flight_1, flight_2, event_1, event_2, c)


def score_candidate(
    candidate: Candidate,
    events: tuple[DopplerEvent, DopplerEvent],
    t0_jitter_s: float = 5.0,
    t0_jitter_steps: int = 21,
    c: float = SPEED_OF_SOUND_MPS,
) -> ScoredCandidate:
    """Refine T0 for this candidate (coarse scan over `t0_jitter_s`
    around a fresh per-point seed, then continuous bounded minimization
    around the best coarse sample) and score by mean squared Doppler
    frequency residual, negated so higher is better."""
    event_1, event_2 = events
    t0_seed = _seed_t0(candidate.point, candidate.flight_1, candidate.flight_2, event_1, event_2, c)

    coarse_deltas = np.linspace(-t0_jitter_s, t0_jitter_s, t0_jitter_steps)
    coarse_costs = [
        _residual_cost_delta(delta, t0_seed, candidate.point, candidate.flight_1, candidate.flight_2, event_1, event_2, c)
        for delta in coarse_deltas
    ]
    best_idx = int(np.argmin(coarse_costs))
    bracket_lo = coarse_deltas[max(best_idx - 1, 0)]
    bracket_hi = coarse_deltas[min(best_idx + 1, len(coarse_deltas) - 1)]

    result = minimize_scalar(
        _residual_cost_delta,
        bounds=(bracket_lo, bracket_hi),
        method="bounded",
        args=(t0_seed, candidate.point, candidate.flight_1, candidate.flight_2, event_1, event_2, c),
    )
    t0_refined = t0_seed + float(result.x)

    modeled_curve_1 = model_doppler_curve(
        candidate.point, candidate.flight_1, event_1.f0_estimate, event_1.curve_times + t0_refined, c
    )
    modeled_curve_2 = model_doppler_curve(
        candidate.point, candidate.flight_2, event_2.f0_estimate, event_2.curve_times + t0_refined, c
    )

    return ScoredCandidate(
        candidate=candidate,
        score=-float(result.fun),
        modeled_curve_1=modeled_curve_1,
        modeled_curve_2=modeled_curve_2,
        t0_refined=t0_refined,
    )


def locate(
    candidates: list[Candidate], events: tuple[DopplerEvent, DopplerEvent], **score_kwargs
) -> ScoredCandidate:
    """Score every candidate and return the single best-scoring one.

    Raises ValueError, rather than letting `max()` fail with an
    unhelpful error or silently returning the first candidate, when
    there is nothing to return a meaningful answer from.
    """
    if not candidates:
        raise ValueError("locate() received no candidates -- did coarse_filter return any survivors?")
    scored = [score_candidate(candidate, events, **score_kwargs) for candidate in candidates]
    if all(np.isinf(s.score) for s in scored):
        raise ValueError("locate() found no candidate with a finite score -- every candidate was unscorable")
    return max(scored, key=lambda s: s.score)
