"""Synthetic test audio: a waveform with one or more baked-in Doppler
curves, generated via the numerical propagation solver + Doppler formula
(acoustic_geolocation.physics), deliberately independent of the
closed-form forward model that acoustic_geolocation.audio.fit uses to
recover parameters from a ridge. Generating with one implementation and
fitting with another gives a genuine cross-check instead of a model
testing itself.

`SynthEvent` takes a generic `(observer_ecef, position_fn, velocity_fn)`
motion, the same shape `physics.propagation.solve_emission_time` and
`types.FlightTrack` already use -- so a real `FlightTrack`'s
`position_ecef`/`velocity_ecef` methods slot in directly for an
end-to-end test that needs the synthesized audio and the ADS-B track fed
to the pipeline to agree on the *same* real-world geometry, not an
abstract flat-frame stand-in. `make_straight_line_event` is a convenience
constructor for the simpler abstract-frame case used by unit tests that
don't need a real FlightTrack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from acoustic_geolocation.physics.constants import SPEED_OF_SOUND_MPS
from acoustic_geolocation.physics.doppler import doppler_shift, radial_velocity
from acoustic_geolocation.physics.propagation import solve_emission_time


@dataclass
class SynthEvent:
    observer_ecef: np.ndarray
    position_fn: Callable[[np.ndarray], np.ndarray]
    velocity_fn: Callable[[np.ndarray], np.ndarray]
    f0: float  # Hz
    t_start: float  # heard-time window to render, recording-relative seconds
    t_end: float


def make_straight_line_event(
    f0: float, v: float, d_min: float, t_cpa_emission: float, t_start: float, t_end: float
) -> SynthEvent:
    """A SynthEvent for a constant-velocity straight-line flyby in an
    abstract flat frame (observer at the origin's `(0, d_min, 0)`, source
    moving along x) -- for unit tests that don't need a real FlightTrack.
    """
    observer_ecef = np.array([0.0, d_min, 0.0])

    def position_fn(t_e: np.ndarray) -> np.ndarray:
        t_e = np.asarray(t_e, dtype=float)
        tau = t_e - t_cpa_emission
        return np.stack([v * tau, np.zeros_like(tau), np.zeros_like(tau)], axis=-1)

    def velocity_fn(t_e: np.ndarray) -> np.ndarray:
        t_e = np.asarray(t_e, dtype=float)
        vel = np.zeros(t_e.shape + (3,))
        vel[..., 0] = v
        return vel

    return SynthEvent(
        observer_ecef=observer_ecef, position_fn=position_fn, velocity_fn=velocity_fn,
        f0=f0, t_start=t_start, t_end=t_end,
    )


def synthesize_waveform(
    events: list[SynthEvent],
    samplerate: int = 8000,
    c: float = SPEED_OF_SOUND_MPS,
    noise_amplitude: float = 0.01,
    reference_distance_m: float = 1000.0,
    seed: int = 0,
    duration_s: float | None = None,
) -> tuple[np.ndarray, int]:
    """`duration_s` overrides the inferred `max(event.t_end)` length --
    needed when events are deliberately clipped to a sub-window shorter
    than the full intended recording (see
    `simulate_recording.py::synthesize_recording_at`), where the overall
    waveform must still span the full recording, not just the loudest
    track's own window.
    """
    duration = duration_s if duration_s is not None else max(event.t_end for event in events)
    n_samples = int(duration * samplerate)
    t_heard = np.arange(n_samples) / samplerate
    waveform = np.zeros(n_samples)

    for event in events:
        mask = (t_heard >= event.t_start) & (t_heard <= event.t_end)
        window_indices = np.where(mask)[0]

        t_e = solve_emission_time(t_heard[window_indices], event.observer_ecef, event.position_fn, event.velocity_fn, c=c)
        valid = ~np.isnan(t_e)
        indices, t_e = window_indices[valid], t_e[valid]

        pos = event.position_fn(t_e)
        vel = event.velocity_fn(t_e)
        v_r = radial_velocity(vel, pos, event.observer_ecef)
        freq = doppler_shift(event.f0, v_r, c)
        distance = np.linalg.norm(event.observer_ecef - pos, axis=-1)
        # Amplitude decays as 1/distance; scaled by a reference distance so
        # the tone sits well above `noise_amplitude` at typical test ranges
        # instead of at some physically "correct" but numerically tiny scale.
        amplitude = reference_distance_m / np.maximum(distance, 1.0)

        phase = 2.0 * np.pi * np.cumsum(freq) / samplerate
        waveform[indices] += amplitude * np.sin(phase)

    rng = np.random.default_rng(seed)
    waveform = waveform + rng.normal(scale=noise_amplitude, size=waveform.shape)
    return waveform, samplerate
