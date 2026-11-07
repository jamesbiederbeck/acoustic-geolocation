"""Fast smoke test for scripts/run_validation_harness.py: a small, fixed-
seed run (a few flights over a few hours, a handful of trials, a tight
per-trial watchdog) that exercises the full generate -> pick-trial ->
synthesize -> geolocate loop end to end.

This does NOT assert the pipeline recovers the truth location -- at this
tiny scale that's genuinely not guaranteed (and asserting it would make
the test flaky for reasons that have nothing to do with a regression in
this file). What it does assert: every trial lands in a known outcome
category (nothing escapes uncaught), and the run completes in bounded
time -- i.e. the harness's own bookkeeping and its `_trial_timeout`
watchdog work, regardless of how the simulated scenario happens to
resolve. Manually running the full-scale harness (see its own docstring)
is how `ok`-rate/accuracy is actually assessed.
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_validation_harness import build_parser, run  # noqa: E402

from acoustic_geolocation.adsb.interface import BBox
from acoustic_geolocation.adsb.local_file import LocalFileAdsbSource
from acoustic_geolocation.physics.geodesy import FlatDem, geodetic_to_ecef
from acoustic_geolocation.pipeline import geolocate
from acoustic_geolocation.types import GridPoint

from .fixtures.simulate_recording import synthesize_recording_at
from .fixtures.synth_tracks import make_straight_track, to_flight_track, write_adsb_csv

T_ANCHOR = 1_700_000_000.0
T0_TRUE = T_ANCHOR - 30.0
TRUTH_LAT, TRUTH_LON, TRUTH_ELEV_M = 47.05, -122.05, 50.0
BBOX = BBox(lat_min=46.95, lon_min=-122.15, lat_max=47.15, lon_max=-121.95)


def test_harness_synthesis_path_recovers_a_controlled_two_flight_scenario(tmp_path):
    """Unlike the random-trial smoke test above, this builds a fixed,
    well-separated two-flight scenario by hand -- the same scenario
    test_pipeline_end_to_end.py already proves the pipeline handles
    quickly and correctly -- but routes the audio through this harness's
    OWN `synthesize_recording_at`/`audible_tracks_in_window` code path,
    not a hand-built SynthEvent list, proving that code path is wired
    correctly, independently of whether the random trial-picker in
    `run()` happens to land on a good trial.
    """
    truth = GridPoint(lat=TRUTH_LAT, lon=TRUTH_LON, elevation_m=TRUTH_ELEV_M, ecef=geodetic_to_ecef(TRUTH_LAT, TRUTH_LON, TRUTH_ELEV_M))

    track1 = make_straight_track(
        "HARNESS1", 47.064, -122.05, 1500.0, heading_deg=90.0, speed_mps=90.0,
        t_anchor=T_ANCHOR, t_start=T_ANCHOR - 300, t_end=T_ANCHOR + 300, is_propeller=True,
    )
    track2 = make_straight_track(
        "HARNESS2", 47.05, -122.034, 1800.0, heading_deg=0.0, speed_mps=100.0,
        t_anchor=T_ANCHOR + 150.0, t_start=T_ANCHOR - 150, t_end=T_ANCHOR + 450, is_propeller=True,
    )
    flight_tracks = [to_flight_track(track1), to_flight_track(track2)]

    adsb_path = tmp_path / "harness_test.csv"
    write_adsb_csv(adsb_path, [track1, track2])

    waveform, samplerate = synthesize_recording_at(
        truth, T0_TRUE, duration_s=600.0, tracks=flight_tracks, audibility_radius_m=5000.0,
        f0_by_icao24={"HARNESS1": 140.0, "HARNESS2": 220.0},
    )
    audio_path = tmp_path / "harness_recording.wav"
    sf.write(str(audio_path), waveform, samplerate)

    recording_date = dt.datetime.fromtimestamp(T0_TRUE, tz=dt.timezone.utc).date()
    result = geolocate(
        audio_path, recording_date, BBOX, LocalFileAdsbSource(adsb_path),
        dem=FlatDem(TRUTH_ELEV_M), coarse_resolution_m=200.0, fine_resolution_m=100.0,
        audibility_radius_m=5000.0,
    )

    lat_diff_m = (result.candidate.point.lat - TRUTH_LAT) * 111_320.0
    lon_diff_m = (result.candidate.point.lon - TRUTH_LON) * 111_320.0 * np.cos(np.radians(TRUTH_LAT))
    distance_m = float(np.hypot(lat_diff_m, lon_diff_m))

    assert distance_m < 150.0
    assert abs(result.t0_refined - T0_TRUE) < 1.0


def test_harness_smoke_run_categorizes_every_trial():
    n_trials = 4
    args = build_parser().parse_args(
        [
            "--n-flights", "30",
            "--n-trials", str(n_trials),
            "--days", "0.25",
            "--date", "2026-06-15",
            "--search-radius-m", "3000",
            "--coarse-resolution-m", "300",
            "--fine-resolution-m", "150",
            "--trial-timeout-s", "8",
            "--seed", "1",
        ]
    )

    start = time.time()
    outcomes = run(args)
    elapsed = time.time() - start

    assert sum(len(records) for records in outcomes.values()) == n_trials
    assert outcomes["exception"] == [], f"unexpected exception(s): {outcomes['exception']}"
    assert elapsed < 90.0, f"smoke-scale harness run took {elapsed:.1f}s, expected well under 90s"
