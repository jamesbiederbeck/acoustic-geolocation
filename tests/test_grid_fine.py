"""Tests for grid/fine.py: Doppler curve synthesis, T0 refinement, and
that scoring + argmax over a fine grid lands on the true observer
location within one fine-grid cell.

Ground truth here is baked directly from model_doppler_curve() itself
(the same forward model score_candidate uses), rather than independently
re-derived -- consistent with test_grid_coarse.py: physics/propagation's
correctness is already independently verified elsewhere
(test_physics_propagation.py's closed-form oracle), so this file tests
grid/fine's own search/scoring logic, not the underlying physics.
"""

import numpy as np

import pytest

from acoustic_geolocation.grid.coarse import delayed_cpa
from acoustic_geolocation.grid.fine import locate, model_doppler_curve, score_candidate, upscale_grid
from acoustic_geolocation.physics.geodesy import FlatDem, geodetic_to_ecef
from acoustic_geolocation.types import Candidate, DopplerEvent, FlightTrack, GridPoint, TrackSample

from .fixtures.synth_tracks import make_straight_track

T_ANCHOR = 1_700_000_000.0
T0_TRUE = T_ANCHOR - 30.0
TRUTH_LAT, TRUTH_LON, TRUTH_ELEV_M = 47.05, -122.05, 50.0
COARSE_RESOLUTION_M = 500.0
FINE_RESOLUTION_M = 100.0
C = 343.0


def _flight_track_from_synth(synth_track) -> FlightTrack:
    samples = tuple(
        TrackSample(t=float(t), lat=float(lat), lon=float(lon), alt_m=float(alt))
        for t, lat, lon, alt in zip(synth_track.times, synth_track.lat, synth_track.lon, synth_track.alt_m)
    )
    return FlightTrack(
        icao24=synth_track.icao24,
        callsign=synth_track.callsign,
        aircraft_type=synth_track.aircraft_type,
        is_propeller=synth_track.is_propeller,
        samples=samples,
    )


def _truth_point() -> GridPoint:
    ecef = geodetic_to_ecef(TRUTH_LAT, TRUTH_LON, TRUTH_ELEV_M)
    return GridPoint(lat=TRUTH_LAT, lon=TRUTH_LON, elevation_m=TRUTH_ELEV_M, ecef=ecef)


def _make_tracks():
    track1 = _flight_track_from_synth(
        make_straight_track(
            "PROP1", 47.064, -122.05, 1500.0, heading_deg=90.0, speed_mps=90.0,
            t_anchor=T_ANCHOR, t_start=T_ANCHOR - 300, t_end=T_ANCHOR + 300, is_propeller=True,
        )
    )
    track2 = _flight_track_from_synth(
        make_straight_track(
            "PROP2", 47.05, -122.034, 1800.0, heading_deg=0.0, speed_mps=100.0,
            t_anchor=T_ANCHOR + 150.0, t_start=T_ANCHOR - 150, t_end=T_ANCHOR + 450, is_propeller=True,
        )
    )
    return track1, track2


def _build_event(event_id, truth, flight, t_heard_center, f0):
    rel_offsets = np.arange(-15.0, 15.0 + 1e-9, 1.0)
    curve_times = (t_heard_center - T0_TRUE) + rel_offsets  # recording-relative seconds
    curve_freqs = model_doppler_curve(truth, flight, f0, curve_times + T0_TRUE, c=C)
    _, d_min = delayed_cpa(truth, flight, C)
    return DopplerEvent(
        event_id=event_id,
        t_heard_start=float(curve_times.min()), t_heard_end=float(curve_times.max()),
        t_heard_ref=float(t_heard_center - T0_TRUE), t_heard_ref_err=0.05,
        f0_estimate=f0, f0_estimate_err=1.0,
        v_estimate=90.0, v_estimate_err=5.0,
        d_min_estimate=d_min, d_min_estimate_err=5.0,
        curve_times=curve_times, curve_freqs=curve_freqs,
    )


def _make_true_events(track1, track2):
    truth = _truth_point()
    t_heard_1, _ = delayed_cpa(truth, track1)
    t_heard_2, _ = delayed_cpa(truth, track2)
    event_1 = _build_event(0, truth, track1, t_heard_1, 400.0)
    event_2 = _build_event(1, truth, track2, t_heard_2, 500.0)
    return event_1, event_2


def test_model_doppler_curve_matches_its_own_ground_truth_exactly():
    track1, track2 = _make_tracks()
    event_1, _ = _make_true_events(track1, track2)
    truth = _truth_point()

    modeled = model_doppler_curve(truth, track1, event_1.f0_estimate, event_1.curve_times + T0_TRUE, c=C)
    np.testing.assert_allclose(modeled, event_1.curve_freqs, atol=1e-6)


def test_score_candidate_recovers_true_t0_at_truth_point():
    track1, track2 = _make_tracks()
    event_1, event_2 = _make_true_events(track1, track2)
    truth = _truth_point()

    candidate = Candidate(
        point=truth, flight_1=track1, flight_2=track2, t0_hypothesis=T0_TRUE, coarse_time_offset_error=0.0
    )
    scored = score_candidate(candidate, (event_1, event_2), c=C)

    assert abs(scored.t0_refined - T0_TRUE) < 0.05
    assert scored.score > -1e-6  # near-zero cost at the exact truth and true T0


def test_truth_point_scores_better_than_a_displaced_point():
    track1, track2 = _make_tracks()
    event_1, event_2 = _make_true_events(track1, track2)
    truth = _truth_point()

    displaced_ecef = geodetic_to_ecef(TRUTH_LAT + 0.01, TRUTH_LON + 0.01, TRUTH_ELEV_M)  # ~1km+ away
    displaced = GridPoint(lat=TRUTH_LAT + 0.01, lon=TRUTH_LON + 0.01, elevation_m=TRUTH_ELEV_M, ecef=displaced_ecef)

    truth_candidate = Candidate(
        point=truth, flight_1=track1, flight_2=track2, t0_hypothesis=T0_TRUE, coarse_time_offset_error=0.0
    )
    displaced_candidate = Candidate(
        point=displaced, flight_1=track1, flight_2=track2, t0_hypothesis=T0_TRUE, coarse_time_offset_error=0.0
    )

    truth_scored = score_candidate(truth_candidate, (event_1, event_2), c=C)
    displaced_scored = score_candidate(displaced_candidate, (event_1, event_2), c=C)

    assert truth_scored.score > displaced_scored.score


def test_argmax_over_fine_grid_recovers_truth_within_one_cell():
    track1, track2 = _make_tracks()
    event_1, event_2 = _make_true_events(track1, track2)
    truth = _truth_point()

    # Use the exact truth point as the coarse "survivor" handed to
    # upscale_grid -- its own [-r/2, r/2] tiling, not an exact match, is
    # what must land a fine point near truth; the test exercises whether
    # argmax over that tiling correctly discriminates the true cell from
    # its many neighbors, not whether truth happens to be a grid point.
    coarse_candidate = Candidate(
        point=truth, flight_1=track1, flight_2=track2, t0_hypothesis=T0_TRUE, coarse_time_offset_error=0.0
    )
    fine_candidates = upscale_grid([coarse_candidate], COARSE_RESOLUTION_M, FINE_RESOLUTION_M, FlatDem(TRUTH_ELEV_M))
    assert len(fine_candidates) > 1

    scored = [score_candidate(candidate, (event_1, event_2), c=C) for candidate in fine_candidates]
    best = max(scored, key=lambda s: s.score)

    lat_diff_m = (best.candidate.point.lat - TRUTH_LAT) * 111_320.0
    lon_diff_m = (best.candidate.point.lon - TRUTH_LON) * 111_320.0 * np.cos(np.radians(TRUTH_LAT))
    distance_m = float(np.hypot(lat_diff_m, lon_diff_m))
    assert distance_m < FINE_RESOLUTION_M * 1.5


def test_locate_raises_on_no_candidates():
    track1, track2 = _make_tracks()
    event_1, event_2 = _make_true_events(track1, track2)
    with pytest.raises(ValueError):
        locate([], (event_1, event_2))
