"""Synthetic test audio: a waveform with one or more baked-in Doppler
curves, generated via the numerical propagation solver + Doppler formula
(acoustic_geolocation.physics), deliberately independent of the
closed-form forward model that acoustic_geolocation.audio.fit uses to
recover parameters from a ridge. Generating with one implementation and
fitting with another gives a genuine cross-check instead of a model
testing itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acoustic_geolocation.physics.constants import SPEED_OF_SOUND_MPS
from acoustic_geolocation.physics.doppler import doppler_shift, radial_velocity
from acoustic_geolocation.physics.propagation import solve_emission_time


@dataclass
class SynthEvent:
    f0: float  # Hz
    v: float  # m/s
    d_min: float  # m
    t_cpa_emission: float  # emission time (recording-relative seconds) of closest (non-delayed) approach
    t_start: float  # heard-time window to render, seconds
    t_end: float


def _straight_line_motion(v: float, t_cpa_emission: float, d_min: float):
    observer = np.array([0.0, d_min, 0.0])

    def position_fn(t_e: np.ndarray) -> np.ndarray:
        t_e = np.asarray(t_e, dtype=float)
        tau = t_e - t_cpa_emission
        return np.stack([v * tau, np.zeros_like(tau), np.zeros_like(tau)], axis=-1)

    def velocity_fn(t_e: np.ndarray) -> np.ndarray:
        t_e = np.asarray(t_e, dtype=float)
        vel = np.zeros(t_e.shape + (3,))
        vel[..., 0] = v
        return vel

    return observer, position_fn, velocity_fn


def synthesize_waveform(
    events: list[SynthEvent],
    samplerate: int = 8000,
    c: float = SPEED_OF_SOUND_MPS,
    noise_amplitude: float = 0.01,
    reference_distance_m: float = 1000.0,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    duration = max(event.t_end for event in events)
    n_samples = int(duration * samplerate)
    t_heard = np.arange(n_samples) / samplerate
    waveform = np.zeros(n_samples)

    for event in events:
        observer, position_fn, velocity_fn = _straight_line_motion(event.v, event.t_cpa_emission, event.d_min)
        mask = (t_heard >= event.t_start) & (t_heard <= event.t_end)
        window_indices = np.where(mask)[0]

        t_e = solve_emission_time(t_heard[window_indices], observer, position_fn, velocity_fn, c=c)
        valid = ~np.isnan(t_e)
        indices, t_e = window_indices[valid], t_e[valid]

        pos = position_fn(t_e)
        vel = velocity_fn(t_e)
        v_r = radial_velocity(vel, pos, observer)
        freq = doppler_shift(event.f0, v_r, c)
        distance = np.linalg.norm(observer - pos, axis=-1)
        # Amplitude decays as 1/distance; scaled by a reference distance so
        # the tone sits well above `noise_amplitude` at typical test ranges
        # instead of at some physically "correct" but numerically tiny scale.
        amplitude = reference_distance_m / np.maximum(distance, 1.0)

        phase = 2.0 * np.pi * np.cumsum(freq) / samplerate
        waveform[indices] += amplitude * np.sin(phase)

    rng = np.random.default_rng(seed)
    waveform = waveform + rng.normal(scale=noise_amplitude, size=waveform.shape)
    return waveform, samplerate
