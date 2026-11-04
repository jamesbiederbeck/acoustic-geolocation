"""Coarse grid filtering: generate a geospatial grid, compute each
candidate point's delayed closest-point-of-approach (CPA) to each
candidate flight pair, and keep points whose predicted inter-event timing
and minimum distance match what the audio stage measured, and for which
no other tracked aircraft would have been audible between the two
events.

The delayed CPA is computed in closed form: minimize slant distance over
emission time per linear track segment (a quadratic in the segment
parameter, clipped to the segment) and take the global minimum across
segments, vectorized over all segments at once (never a per-segment
Python loop -- a grid search calls this once per point per track, so
that loop would make "coarse" anything but cheap). This never calls
physics.propagation.solve_emission_time's iterative solver -- heard-time
is monotonic in emission-time for subsonic sources, so there is exactly
one minimum-distance emission time and no implicit equation to invert
here. That solver is reserved for grid/fine.py, which needs emission
time at arbitrary heard-times, not just at the CPA.

The silence filter (`_anything_else_audible_between`) checks audibility
across the *entire* emission-time window implied by the two events'
heard times, not just at each candidate distractor's own CPA -- a
distractor whose true closest approach lands just outside the window can
still be well within the audibility radius for part of it.

**Known v1 limitation**: `delayed_cpa` returns the single global closest
approach for a track. An aircraft flying a repeating pattern (e.g. a
traffic-pattern circuit) that passes the same point more than once is
only checked at its closest overall pass, not every pass.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..adsb.interface import BBox
from ..physics.constants import SPEED_OF_SOUND_MPS
from ..physics.geodesy import DemSource, geodetic_to_ecef
from ..types import Candidate, DopplerEvent, FlightTrack, GridPoint

_METERS_PER_DEGREE_LAT = 111_320.0


def generate_grid(bbox: BBox, resolution_m: float, dem: DemSource) -> list[GridPoint]:
    lat_mid = 0.5 * (bbox.lat_min + bbox.lat_max)
    meters_per_degree_lon = _METERS_PER_DEGREE_LAT * max(np.cos(np.radians(lat_mid)), 1e-6)

    lat_step = resolution_m / _METERS_PER_DEGREE_LAT
    lon_step = resolution_m / meters_per_degree_lon

    lats = np.arange(bbox.lat_min, bbox.lat_max + lat_step / 2.0, lat_step)
    lons = np.arange(bbox.lon_min, bbox.lon_max + lon_step / 2.0, lon_step)

    points = []
    for lat in lats:
        for lon in lons:
            elevation_m = float(dem.elevation_m(float(lat), float(lon)))
            ecef = geodetic_to_ecef(lat, lon, elevation_m)
            points.append(GridPoint(lat=float(lat), lon=float(lon), elevation_m=elevation_m, ecef=ecef))
    return points


def _segments_with_margin(track: FlightTrack) -> tuple[np.ndarray, np.ndarray]:
    """Sample times/positions extended by `extrapolation_margin_s` at each
    end (linearly, using the boundary segment's velocity), so the CPA
    search also covers the bounded-extrapolation region FlightTrack
    itself supports -- a closest approach can fall just past a track's
    last sample."""
    times = track.sample_times
    positions = track.sample_positions_ecef
    margin = track.extrapolation_margin_s

    v_start = (positions[1] - positions[0]) / (times[1] - times[0])
    v_end = (positions[-1] - positions[-2]) / (times[-1] - times[-2])

    all_times = np.concatenate([[times[0] - margin], times, [times[-1] + margin]])
    all_positions = np.vstack([positions[0] - v_start * margin, positions, positions[-1] + v_end * margin])
    return all_times, all_positions


def delayed_cpa(point: GridPoint, track: FlightTrack, c: float = SPEED_OF_SOUND_MPS) -> tuple[float, float]:
    """Return (t_heard, d_min): the heard time and slant distance of
    `track`'s closest (emission-time) approach to `point`.

    Minimizes distance over emission time per segment -- s_star = argmin
    over s in [0, 1] of |observer - (p0 + (p1 - p0) * s)| -- across all
    segments at once (vectorized, not a Python-level loop), then takes
    the global minimum. Keeping this vectorized matters: a grid search
    calls this once per (point, track) pair, so a per-segment Python loop
    here would make "coarse" anything but cheap.
    """
    times, positions = _segments_with_margin(track)
    p0, p1 = positions[:-1], positions[1:]
    t0, t1 = times[:-1], times[1:]

    direction = p1 - p0
    denom = np.sum(direction * direction, axis=1)
    denom_safe = np.where(denom > 1e-9, denom, 1.0)
    s_star = np.where(denom > 1e-9, np.sum((point.ecef - p0) * direction, axis=1) / denom_safe, 0.0)
    s_star = np.clip(s_star, 0.0, 1.0)

    position_star = p0 + direction * s_star[:, None]
    distance = np.linalg.norm(point.ecef - position_star, axis=1)
    t_emission = t0 + s_star * (t1 - t0)

    best_idx = int(np.argmin(distance))
    best_t_e, best_d = float(t_emission[best_idx]), float(distance[best_idx])
    return best_t_e + best_d / c, best_d


def _default_time_margin_s(resolution_m: float, event_1: DopplerEvent, event_2: DopplerEvent, c: float) -> float:
    """Minimum sensible coarse time-offset tolerance: how much a grid
    point's predicted inter-event offset can shift just from being
    `resolution_m / sqrt(2)` (worst-case grid quantization) away from the
    true location, plus 3 sigma of the audio stage's own timing error.
    Too tight a margin rejects the true grid cell outright.
    """
    spatial_offset_m = resolution_m / np.sqrt(2.0)
    v1 = max(event_1.v_estimate, 1.0)
    v2 = max(event_2.v_estimate, 1.0)
    timing_err_s = float(np.hypot(event_1.t_heard_ref_err, event_2.t_heard_ref_err))
    return spatial_offset_m * (1.0 / v1 + 1.0 / v2 + 2.0 / c) + 3.0 * timing_err_s


def _default_distance_tolerance_m(resolution_m: float) -> float:
    return resolution_m / np.sqrt(2.0)


def _min_distance_in_heard_window(
    point: GridPoint, track: FlightTrack, t_heard_window: tuple[float, float], c: float
) -> float | None:
    """Minimum slant distance from `point` to `track`, restricted to the
    portion of the track whose *heard* time at `point` falls inside
    `t_heard_window`. Returns None if the track is never heard at `point`
    at all during that window.

    This is deliberately not just `delayed_cpa` filtered to the window:
    a track's single global closest approach can land just outside the
    window while the track is still well within the audibility radius
    for part of the window -- that case must still count as audible.

    Heard-time at a fixed point is a strictly increasing function of
    emission time for any subsonic track (derivative 1 - v_r/c > 0), so
    the per-segment endpoint heard-times are monotonic and the heard-time
    window maps to a well-defined emission-time window via interpolation.
    """
    times, positions = _segments_with_margin(track)
    heard_times = times + np.linalg.norm(point.ecef - positions, axis=1) / c

    t_lo, t_hi = sorted(t_heard_window)
    if t_hi < heard_times[0] or t_lo > heard_times[-1]:
        return None  # window entirely outside the track's heard-time span

    t_emission_lo, t_emission_hi = np.interp([t_lo, t_hi], heard_times, times)

    p0, p1 = positions[:-1], positions[1:]
    t0, t1 = times[:-1], times[1:]
    direction = p1 - p0
    denom = np.sum(direction * direction, axis=1)
    denom_safe = np.where(denom > 1e-9, denom, 1.0)
    s_unconstrained = np.where(denom > 1e-9, np.sum((point.ecef - p0) * direction, axis=1) / denom_safe, 0.0)

    # Clip each segment's search to its overlap with [t_emission_lo, t_emission_hi].
    seg_s_lo = np.clip((t_emission_lo - t0) / (t1 - t0), 0.0, 1.0)
    seg_s_hi = np.clip((t_emission_hi - t0) / (t1 - t0), 0.0, 1.0)
    has_overlap = (t1 > t_emission_lo) & (t0 < t_emission_hi)
    if not np.any(has_overlap):
        return None

    s_star = np.clip(s_unconstrained, np.minimum(seg_s_lo, seg_s_hi), np.maximum(seg_s_lo, seg_s_hi))
    position_star = p0 + direction * s_star[:, None]
    distance = np.where(has_overlap, np.linalg.norm(point.ecef - position_star, axis=1), np.inf)

    return float(np.min(distance))


def _anything_else_audible_between(
    point: GridPoint,
    silence_tracks: list[FlightTrack],
    excluded: tuple[FlightTrack, FlightTrack],
    t_window: tuple[float, float],
    audibility_radius_m: float,
    c: float,
) -> bool:
    excluded_ids = {id(track) for track in excluded}
    for track in silence_tracks:
        if id(track) in excluded_ids:
            continue
        min_distance = _min_distance_in_heard_window(point, track, t_window, c)
        if min_distance is not None and min_distance <= audibility_radius_m:
            return True
    return False


def coarse_filter(
    points: list[GridPoint],
    events: tuple[DopplerEvent, DopplerEvent],
    match_tracks: list[FlightTrack],
    silence_tracks: list[FlightTrack],
    resolution_m: float,
    error_margin_s: float | None = None,
    d_min_tolerance_m: float | None = None,
    audibility_radius_m: float = 5000.0,
    c: float = SPEED_OF_SOUND_MPS,
) -> list[Candidate]:
    """Keep (point, flight_1, flight_2) triples whose predicted delayed-CPA
    timing and distance match the two detected audio events, and for
    which the silence filter finds nothing else audible between them.

    `match_tracks` should already be narrowed to aircraft types whose
    acoustic signature is modeled (via adsb/filters.py); `silence_tracks`
    should be *all* tracks in range, matching or not, since any audible
    aircraft -- not just matching ones -- can rule out a candidate point.
    """
    event_1, event_2 = events
    measured_offset_s = event_2.t_heard_ref - event_1.t_heard_ref
    time_margin_s = (
        error_margin_s if error_margin_s is not None else _default_time_margin_s(resolution_m, event_1, event_2, c)
    )
    distance_tolerance_m = (
        d_min_tolerance_m if d_min_tolerance_m is not None else _default_distance_tolerance_m(resolution_m)
    )

    candidates: list[Candidate] = []
    for point in points:
        for flight_1, flight_2 in itertools.permutations(match_tracks, 2):
            t_heard_1, d_min_1 = delayed_cpa(point, flight_1, c)
            t_heard_2, d_min_2 = delayed_cpa(point, flight_2, c)

            predicted_offset_s = t_heard_2 - t_heard_1
            offset_error = predicted_offset_s - measured_offset_s
            if abs(offset_error) > time_margin_s:
                continue

            if abs(d_min_1 - event_1.d_min_estimate) > distance_tolerance_m + 3.0 * event_1.d_min_estimate_err:
                continue
            if abs(d_min_2 - event_2.d_min_estimate) > distance_tolerance_m + 3.0 * event_2.d_min_estimate_err:
                continue

            if _anything_else_audible_between(
                point, silence_tracks, (flight_1, flight_2), (t_heard_1, t_heard_2), audibility_radius_m, c
            ):
                continue

            candidates.append(
                Candidate(
                    point=point,
                    flight_1=flight_1,
                    flight_2=flight_2,
                    t0_hypothesis=t_heard_1 - event_1.t_heard_ref,
                    coarse_time_offset_error=offset_error,
                )
            )

    return candidates
