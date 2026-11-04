"""Integration test: coarse_filter -> upscale_grid -> locate on the same
two-non-parallel-track scenario as test_grid_coarse.py, with
recording-relative event times (not absolute epoch times) as the real
audio stage would produce.

This is the one test that exercises three things the per-stage unit
tests don't: the coarse-to-fine hand-off itself, whether the known
coarse-stage mirror-image ambiguity (see test_grid_coarse.py) actually
gets resolved by fine-grid scoring rather than just asserted away, and
the end-to-end wall time grid/fine's current per-candidate cost implies
for commit 7's real-data scale.
"""

import time

import numpy as np

from acoustic_geolocation.adsb.interface import BBox
from acoustic_geolocation.grid.coarse import coarse_filter, delayed_cpa, generate_grid
from acoustic_geolocation.grid.fine import locate, model_doppler_curve, upscale_grid
from acoustic_geolocation.physics.geodesy import FlatDem, geodetic_to_ecef
from acoustic_geolocation.types import DopplerEvent, FlightTrack, GridPoint, TrackSample

from .fixtures.synth_tracks import make_straight_track

T_ANCHOR = 1_700_000_000.0
T0_TRUE = T_ANCHOR - 30.0
TRUTH_LAT, TRUTH_LON, TRUTH_ELEV_M = 47.05, -122.05, 50.0
BBOX = BBox(lat_min=46.95, lon_min=-122.15, lat_max=47.15, lon_max=-121.95)
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
    return GridPoint(lat=TRUTH_LAT, lon=TRUTH_LON, elevation_m=TRUTH_ELEV_M, ecef=geodetic_to_ecef(TRUTH_LAT, TRUTH_LON, TRUTH_ELEV_M))


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


def _make_events(track1, track2):
    """Build DopplerEvents with *real* curve_times/curve_freqs (not
    placeholders): fine-grid scoring compares modeled curves against
    these pointwise, so a meaningful test of whether scoring actually
    discriminates truth from the coarse-stage mirror candidate needs
    real ridge samples, not a single dummy point.
    """
    truth = _truth_point()
    t_heard_1, d_min_1 = delayed_cpa(truth, track1)
    t_heard_2, d_min_2 = delayed_cpa(truth, track2)
    rel_offsets = np.arange(-15.0, 15.0 + 1e-9, 1.0)

    def build(event_id, flight, t_heard_center, f0, d_min):
        curve_times = (t_heard_center - T0_TRUE) + rel_offsets  # recording-relative
        curve_freqs = model_doppler_curve(truth, flight, f0, curve_times + T0_TRUE, c=C)
        return DopplerEvent(
            event_id=event_id,
            t_heard_start=float(curve_times.min()), t_heard_end=float(curve_times.max()),
            t_heard_ref=float(t_heard_center - T0_TRUE), t_heard_ref_err=0.05,
            f0_estimate=f0, f0_estimate_err=1.0,
            v_estimate=90.0, v_estimate_err=5.0,
            d_min_estimate=d_min, d_min_estimate_err=5.0,
            curve_times=curve_times, curve_freqs=curve_freqs,
        )

    event_1 = build(0, track1, t_heard_1, 400.0, d_min_1)
    event_2 = build(1, track2, t_heard_2, 500.0, d_min_2)
    return event_1, event_2


def _distance_m(lat1, lon1, lat2, lon2) -> float:
    lat_mid = np.radians(0.5 * (lat1 + lat2))
    dlat_m = (lat1 - lat2) * 111_320.0
    dlon_m = (lon1 - lon2) * 111_320.0 * np.cos(lat_mid)
    return float(np.hypot(dlat_m, dlon_m))


def test_coarse_to_fine_pipeline_recovers_truth_over_the_mirror_candidate():
    track1, track2 = _make_tracks()
    event_1, event_2 = _make_events(track1, track2)

    start = time.monotonic()
    points = generate_grid(BBOX, COARSE_RESOLUTION_M, FlatDem(TRUTH_ELEV_M))
    coarse_survivors = coarse_filter(
        points, (event_1, event_2), match_tracks=[track1, track2], silence_tracks=[track1, track2],
        resolution_m=COARSE_RESOLUTION_M,
    )
    print(f"coarse survivors: {len(coarse_survivors)}")
    for survivor in coarse_survivors:
        print(f"  {survivor.point.lat:.4f}, {survivor.point.lon:.4f}")

    fine_candidates = upscale_grid(coarse_survivors, COARSE_RESOLUTION_M, FINE_RESOLUTION_M, FlatDem(TRUTH_ELEV_M))
    print(f"fine candidates: {len(fine_candidates)}")

    best = locate(fine_candidates, (event_1, event_2))
    elapsed = time.monotonic() - start
    print(f"best: {best.candidate.point.lat:.5f}, {best.candidate.point.lon:.5f}, score={best.score:.4f}, t0_err={best.t0_refined - T0_TRUE:.3f}s")
    print(f"wall time: {elapsed:.2f}s")

    assert _distance_m(best.candidate.point.lat, best.candidate.point.lon, TRUTH_LAT, TRUTH_LON) < 150.0
    assert abs(best.t0_refined - T0_TRUE) < 0.5
