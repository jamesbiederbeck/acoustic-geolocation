"""End-to-end check of the audio stage: spectrogram -> ridge extraction ->
closed-form fit, run on synthetic audio generated via the independent
numerical solver (tests/fixtures/synth_audio.py).
"""

import numpy as np
import soundfile as sf

from acoustic_geolocation.audio.events import find_doppler_curves
from acoustic_geolocation.audio.fit import fit_doppler_event
from acoustic_geolocation.audio.spectrogram import generate_spectrogram
from acoustic_geolocation.physics.constants import SPEED_OF_SOUND_MPS

from .fixtures.synth_audio import make_straight_line_event, synthesize_waveform

SAMPLERATE = 8000


def test_recovers_single_event(tmp_path):
    event = make_straight_line_event(f0=400.0, v=90.0, d_min=1500.0, t_cpa_emission=10.0, t_start=0.0, t_end=20.0)
    waveform, sr = synthesize_waveform([event], samplerate=SAMPLERATE)
    path = tmp_path / "single.wav"
    sf.write(str(path), waveform, sr)

    spec = generate_spectrogram(path, nperseg=1024)
    ridges = find_doppler_curves(spec)
    assert len(ridges) == 1

    fitted = fit_doppler_event(ridges[0], event_id=0)

    np.testing.assert_allclose(fitted.f0_estimate, event.f0, rtol=0.005)

    true_t_heard_ref = 10.0 + 1500.0 / SPEED_OF_SOUND_MPS  # t_cpa_emission + d_min / c
    assert abs(fitted.t_heard_ref - true_t_heard_ref) < 0.2

    # v and d_min trade off against each other when the ridge doesn't reach
    # its asymptotes -- that's real degeneracy in the physics, not a bug --
    # so check the truth is within the fit's own reported uncertainty.
    assert abs(fitted.v_estimate - 90.0) < max(3 * fitted.v_estimate_err, 20.0)
    assert abs(fitted.d_min_estimate - 1500.0) < max(3 * fitted.d_min_estimate_err, 500.0)


def test_recovers_two_time_separated_events_in_order(tmp_path):
    event1 = make_straight_line_event(f0=300.0, v=80.0, d_min=1200.0, t_cpa_emission=10.0, t_start=0.0, t_end=20.0)
    event2 = make_straight_line_event(f0=500.0, v=100.0, d_min=1800.0, t_cpa_emission=50.0, t_start=40.0, t_end=60.0)
    waveform, sr = synthesize_waveform([event1, event2], samplerate=SAMPLERATE)
    path = tmp_path / "two.wav"
    sf.write(str(path), waveform, sr)

    spec = generate_spectrogram(path, nperseg=1024)
    ridges = find_doppler_curves(spec)
    assert len(ridges) == 2

    fitted = [fit_doppler_event(ridge, event_id=i) for i, ridge in enumerate(ridges)]
    assert fitted[0].t_heard_ref < fitted[1].t_heard_ref
    np.testing.assert_allclose(fitted[0].f0_estimate, event1.f0, rtol=0.01)
    np.testing.assert_allclose(fitted[1].f0_estimate, event2.f0, rtol=0.01)
