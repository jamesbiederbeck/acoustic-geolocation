"""Ridge extraction and Doppler-event segmentation from a spectrogram.

Per-frame peak frequency is found via parabolic sub-bin interpolation on
log-magnitude (plain per-bin peak picking quantizes the ridge to the FFT
bin width, which is too coarse for Doppler-curve fitting). An SNR
threshold against the per-bin noise floor suppresses frames with no
signal. Ridges are then built by greedy nearest-frequency continuation
across frames, tolerating a bounded per-frame frequency jump and a short
run of missing/rejected frames, so one continuous Doppler curve isn't
accidentally split by a dropout and two unrelated curves aren't
accidentally merged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectrogram import Spectrogram


@dataclass(eq=False)
class Ridge:
    times: np.ndarray  # (N,) seconds
    freqs: np.ndarray  # (N,) Hz


def _parabolic_peak_freq(column_db: np.ndarray, freqs: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(column_db) - 1:
        return float(freqs[idx])
    y0, y1, y2 = column_db[idx - 1], column_db[idx], column_db[idx + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(freqs[idx])
    delta = 0.5 * (y0 - y2) / denom
    bin_spacing = freqs[idx + 1] - freqs[idx]
    return float(freqs[idx] + delta * bin_spacing)


def find_doppler_curves(
    spec: Spectrogram,
    snr_threshold_db: float = 18.0,
    max_jump_hz_per_frame: float = 40.0,
    max_gap_frames: int = 2,
    min_ridge_frames: int = 5,
) -> list[Ridge]:
    magnitude_db = 20.0 * np.log10(np.abs(spec.Sxx) + 1e-12)
    noise_floor_db = np.median(magnitude_db, axis=1)

    peak_freq_per_frame = np.full(magnitude_db.shape[1], np.nan)
    for frame_idx in range(magnitude_db.shape[1]):
        column = magnitude_db[:, frame_idx]
        bin_idx = int(np.argmax(column))
        if column[bin_idx] - noise_floor_db[bin_idx] < snr_threshold_db:
            continue
        peak_freq_per_frame[frame_idx] = _parabolic_peak_freq(column, spec.freqs, bin_idx)

    ridges: list[Ridge] = []
    current_times: list[float] = []
    current_freqs: list[float] = []
    gap = 0

    def flush() -> None:
        if len(current_times) >= min_ridge_frames:
            ridges.append(Ridge(times=np.array(current_times), freqs=np.array(current_freqs)))
        current_times.clear()
        current_freqs.clear()

    for frame_idx, freq in enumerate(peak_freq_per_frame):
        is_gap = np.isnan(freq)
        if not is_gap and current_freqs and abs(freq - current_freqs[-1]) > max_jump_hz_per_frame * (gap + 1):
            # Treat a single spurious jump (e.g. a noise spike stealing the
            # argmax for one frame) as a gap rather than ending the ridge --
            # only a *sustained* run of bad frames should split it.
            is_gap = True

        if is_gap:
            gap += 1
            if gap > max_gap_frames:
                flush()
                gap = 0
            continue

        gap = 0
        current_times.append(float(spec.times[frame_idx]))
        current_freqs.append(float(freq))
    flush()

    return ridges
