"""Audibility-aware composite audio synthesis for the validation harness:
given an observer location, a recording window, and a (potentially large)
pool of FlightTracks, decide which aircraft would actually have been
audible and render their Doppler-shifted tones into one waveform.

Audibility is governed by a single `audibility_radius_m` parameter --
deliberately the *same* parameter `grid/coarse.py`'s coarse silence filter
uses, not a second, independently-invented notion of "how far can this be
heard." It is a hard cutoff, not a soft falloff: a track beyond it
contributes nothing at all (matching the coarse filter's own binary
"could not possibly be heard" semantics), while within it,
`synth_audio.py::synthesize_waveform`'s existing `reference_distance_m /
distance` inverse-square-intensity amplitude law governs loudness, using
a fixed `reference_distance_m` tuned for a clean signal-to-noise ratio
(not derived *from* `audibility_radius_m` -- doing that would make
anything near the radius's edge too quiet for `audio/events.py`'s 18dB
SNR ridge-detection threshold to see at all, defeating the point of
including it).
"""

from __future__ import annotations

import numpy as np

from acoustic_geolocation.grid.coarse import _min_distance_in_heard_window
from acoustic_geolocation.physics.constants import SPEED_OF_SOUND_MPS
from acoustic_geolocation.types import FlightTrack, GridPoint

from .synth_audio import SynthEvent, synthesize_waveform


def audible_tracks_in_window(
    observer: GridPoint,
    tracks: list[FlightTrack],
    t_heard_window: tuple[float, float],
    audibility_radius_m: float,
    c: float = SPEED_OF_SOUND_MPS,
) -> list[FlightTrack]:
    """Cheap prefilter: which tracks come within `audibility_radius_m` of
    `observer` at some point during `t_heard_window`, reusing
    `grid/coarse.py`'s own closed-form windowed-minimum-distance check
    (already correct for "audible for part of the window even though its
    own closest approach lands outside it") instead of running the
    expensive Newton solve on every candidate track.
    """
    audible = []
    for track in tracks:
        min_distance = _min_distance_in_heard_window(observer, track, t_heard_window, c)
        if min_distance is not None and min_distance <= audibility_radius_m:
            audible.append(track)
    return audible


def _audible_subwindow(
    observer: GridPoint,
    track: FlightTrack,
    t0: float,
    duration_s: float,
    audibility_radius_m: float,
    sample_dt_s: float = 2.0,
    margin_s: float = 5.0,
) -> tuple[float, float] | None:
    """The recording-relative sub-interval of `[0, duration_s]` during
    which `track` is actually within `audibility_radius_m` of `observer`
    (plus a small margin), not the full recording window.

    Rendering every audible track across the *entire* recording duration
    -- rather than just the stretch where it's actually close -- has two
    costs: `synth_audio.py::solve_emission_time`'s Newton iteration runs
    over the full sample count regardless of how little of it matters
    (observed: ~4s for a 1M-sample heard-time array against one short
    track, worse than linearly for longer arrays), and it needlessly
    keeps a quiet, far-away track "present" for the whole clip, which
    only helps trigger `audio/events.py::find_doppler_curves`'s
    one-peak-per-frame limitation by extending the window in which it
    could get confused with another track. Clipping to the actual
    audible stretch avoids both.
    """
    times_abs = np.arange(t0, t0 + duration_s, sample_dt_s)
    positions = track.position_ecef(times_abs)
    distance = np.linalg.norm(observer.ecef - positions, axis=-1)
    mask = np.isfinite(distance) & (distance <= audibility_radius_m)
    if not np.any(mask):
        return None
    relative = times_abs - t0
    t_start = max(0.0, float(relative[mask].min()) - margin_s)
    t_end = min(duration_s, float(relative[mask].max()) + margin_s)
    return t_start, t_end


def synthesize_recording_at(
    observer: GridPoint,
    t0: float,
    duration_s: float,
    tracks: list[FlightTrack],
    audibility_radius_m: float,
    f0_by_icao24: dict[str, float] | None = None,
    default_f0_hz: float = 120.0,
    reference_distance_m: float = 1000.0,
    samplerate: int = 8000,
    noise_amplitude: float = 0.01,
    c: float = SPEED_OF_SOUND_MPS,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Render a recording-relative waveform of duration `duration_s`
    starting at absolute epoch `t0`, including the Doppler-shifted tone of
    every track that comes within `audibility_radius_m` of `observer`
    during `[t0, t0 + duration_s]`.

    `reference_distance_m` (same default as `synthesize_waveform`'s own)
    keeps amplitude well above the noise floor throughout the audibility
    radius, not just near its center -- `audibility_radius_m` only decides
    *which* tracks are included at all (via the prefilter below), not how
    loud an included one sounds.

    Per-aircraft engine-tone frequency comes from `f0_by_icao24` (keyed by
    the track's `icao24`, stripped of any `LocalFileAdsbSource`-added
    `#<segment>` suffix) -- `FlightTrack` itself carries no such field, so
    the harness passes along the frequencies it sampled when generating
    the world. Tracks with no entry fall back to `default_f0_hz`.
    """
    audible = audible_tracks_in_window(observer, tracks, (t0, t0 + duration_s), audibility_radius_m, c)
    f0_by_icao24 = f0_by_icao24 or {}

    events = []
    for track in audible:
        subwindow = _audible_subwindow(observer, track, t0, duration_s, audibility_radius_m)
        if subwindow is None:
            continue  # prefilter used a coarser check; re-confirm narrows nothing out incorrectly, just skips edge cases
        f0 = f0_by_icao24.get(track.icao24.split("#")[0], default_f0_hz)

        def position_fn(t_rel, track=track):
            return track.position_ecef(np.asarray(t_rel, dtype=float) + t0)

        def velocity_fn(t_rel, track=track):
            return track.velocity_ecef(np.asarray(t_rel, dtype=float) + t0)

        events.append(
            SynthEvent(
                observer_ecef=observer.ecef,
                position_fn=position_fn,
                velocity_fn=velocity_fn,
                f0=f0,
                t_start=subwindow[0],
                t_end=subwindow[1],
            )
        )

    if not events:
        n_samples = int(duration_s * samplerate)
        rng = np.random.default_rng(seed)
        return rng.normal(scale=noise_amplitude, size=n_samples), samplerate

    return synthesize_waveform(
        events, samplerate=samplerate, c=c, noise_amplitude=noise_amplitude,
        reference_distance_m=reference_distance_m, seed=seed, duration_s=duration_s,
    )
