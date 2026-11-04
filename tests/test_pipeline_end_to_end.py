"""Full end-to-end test: synthetic audio (generated from real FlightTrack
ECEF motion, not an abstract flat frame) + a local ADS-B CSV dump of the
same tracks, run through pipeline.geolocate() end to end -- audio
analysis, ADS-B ingestion, coarse filtering, and fine-grid scoring all
exercised together, each with its own real implementation rather than a
hand-constructed DopplerEvent/Candidate as the per-stage tests use.

The two matched flights' audible windows at the truth point are kept
non-overlapping (track1's finishes well before track2's starts) since
audio/events.py::find_doppler_curves takes one frequency peak per time
frame and can't separate two truly overlapping curves -- a known v1
limitation, not something this test is meant to probe.
"""

import datetime as dt

import numpy as np
import pytest
import soundfile as sf

from acoustic_geolocation.adsb.interface import BBox
from acoustic_geolocation.adsb.local_file import LocalFileAdsbSource
from acoustic_geolocation.audio.events import find_doppler_curves
from acoustic_geolocation.audio.fit import fit_doppler_event
from acoustic_geolocation.audio.spectrogram import generate_spectrogram
from acoustic_geolocation.grid.coarse import delayed_cpa
from acoustic_geolocation.physics.geodesy import FlatDem, geodetic_to_ecef
from acoustic_geolocation.pipeline import geolocate
from acoustic_geolocation.types import FlightTrack, GridPoint, TrackSample

from .fixtures.synth_audio import SynthEvent, synthesize_waveform
from .fixtures.synth_tracks import make_straight_track, write_adsb_csv

T_ANCHOR = 1_700_000_000.0
T0_TRUE = T_ANCHOR - 30.0
TRUTH_LAT, TRUTH_LON, TRUTH_ELEV_M = 47.05, -122.05, 50.0
BBOX = BBox(lat_min=46.95, lon_min=-122.15, lat_max=47.15, lon_max=-121.95)


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


def _shifted(flight_method, t0: float):
    """Wrap a FlightTrack.position_ecef/velocity_ecef (absolute epoch
    time) so it instead takes recording-relative time, matching what
    synth_audio.synthesize_waveform's internal heard-time axis uses."""

    def wrapped(t_relative):
        return flight_method(np.asarray(t_relative, dtype=float) + t0)

    return wrapped


@pytest.mark.parametrize(
    "coarse_resolution_m, fine_resolution_m",
    [
        (200.0, 100.0),  # the CLI's own shipped defaults -- must stay covered
        (500.0, 100.0),
    ],
)
def test_pipeline_recovers_truth_from_synthetic_audio_and_adsb_dump(
    tmp_path, coarse_resolution_m, fine_resolution_m
):
    truth_ecef = geodetic_to_ecef(TRUTH_LAT, TRUTH_LON, TRUTH_ELEV_M)
    truth = GridPoint(lat=TRUTH_LAT, lon=TRUTH_LON, elevation_m=TRUTH_ELEV_M, ecef=truth_ecef)

    synth_track1 = make_straight_track(
        "PROP1", 47.064, -122.05, 1500.0, heading_deg=90.0, speed_mps=90.0,
        t_anchor=T_ANCHOR, t_start=T_ANCHOR - 300, t_end=T_ANCHOR + 300, is_propeller=True,
    )
    synth_track2 = make_straight_track(
        "PROP2", 47.05, -122.034, 1800.0, heading_deg=0.0, speed_mps=100.0,
        t_anchor=T_ANCHOR + 150.0, t_start=T_ANCHOR - 150, t_end=T_ANCHOR + 450, is_propeller=True,
    )
    track1 = _flight_track_from_synth(synth_track1)
    track2 = _flight_track_from_synth(synth_track2)

    adsb_path = tmp_path / "adsb.csv"
    write_adsb_csv(adsb_path, [synth_track1, synth_track2])

    t_heard_1, _ = delayed_cpa(truth, track1)
    t_heard_2, _ = delayed_cpa(truth, track2)
    rel_1, rel_2 = t_heard_1 - T0_TRUE, t_heard_2 - T0_TRUE
    assert rel_1 + 10.0 < rel_2 - 10.0, "events must not overlap -- see module docstring"

    audio_event_1 = SynthEvent(
        observer_ecef=truth.ecef,
        position_fn=_shifted(track1.position_ecef, T0_TRUE),
        velocity_fn=_shifted(track1.velocity_ecef, T0_TRUE),
        f0=400.0, t_start=rel_1 - 10.0, t_end=rel_1 + 10.0,
    )
    audio_event_2 = SynthEvent(
        observer_ecef=truth.ecef,
        position_fn=_shifted(track2.position_ecef, T0_TRUE),
        velocity_fn=_shifted(track2.velocity_ecef, T0_TRUE),
        f0=500.0, t_start=rel_2 - 10.0, t_end=rel_2 + 10.0,
    )
    waveform, samplerate = synthesize_waveform([audio_event_1, audio_event_2], samplerate=8000)
    audio_path = tmp_path / "recording.wav"
    sf.write(str(audio_path), waveform, samplerate)

    # Diagnostic only: the fitted d_min is known to be biased low relative
    # to the true delayed-CPA distance (v^2/d_min at the crossing is what
    # the ridge slope actually pins down, not v and d_min independently) --
    # printing both here makes that bias, and whether its own reported
    # error bar covers it, visible if this test ever starts failing.
    spectrogram = generate_spectrogram(audio_path)
    ridges = find_doppler_curves(spectrogram)
    for ridge, true_d_min in zip(ridges, (delayed_cpa(truth, track1)[1], delayed_cpa(truth, track2)[1])):
        fitted = fit_doppler_event(ridge, event_id=0)
        print(
            f"fitted d_min={fitted.d_min_estimate:.1f}m (+-{fitted.d_min_estimate_err:.1f}m) "
            f"vs true {true_d_min:.1f}m"
        )

    recording_date = dt.datetime.fromtimestamp(T0_TRUE, tz=dt.timezone.utc).date()

    result = geolocate(
        audio_path,
        recording_date,
        BBOX,
        adsb_source=LocalFileAdsbSource(adsb_path),
        dem=FlatDem(TRUTH_ELEV_M),
        coarse_resolution_m=coarse_resolution_m,
        fine_resolution_m=fine_resolution_m,
    )

    lat_diff_m = (result.candidate.point.lat - TRUTH_LAT) * 111_320.0
    lon_diff_m = (result.candidate.point.lon - TRUTH_LON) * 111_320.0 * np.cos(np.radians(TRUTH_LAT))
    distance_m = float(np.hypot(lat_diff_m, lon_diff_m))

    print(f"recovered: {result.candidate.point.lat:.5f}, {result.candidate.point.lon:.5f}, "
          f"distance={distance_m:.1f}m, t0_err={result.t0_refined - T0_TRUE:.2f}s, score={result.score:.4f}")

    assert distance_m < 150.0
    assert abs(result.t0_refined - T0_TRUE) < 1.0
