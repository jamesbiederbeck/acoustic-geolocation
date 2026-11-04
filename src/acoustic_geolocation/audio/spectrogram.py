"""STFT-based spectrogram generation from an audio file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft


@dataclass(eq=False)
class Spectrogram:
    freqs: np.ndarray  # (F,) Hz
    times: np.ndarray  # (T,) seconds from recording start
    Sxx: np.ndarray  # (F, T) complex STFT


def generate_spectrogram(path: str | Path, nperseg: int = 4096, noverlap: int | None = None) -> Spectrogram:
    """Load `path` (mono-mixed if multichannel) and compute its STFT."""
    data, samplerate = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if noverlap is None:
        noverlap = (3 * nperseg) // 4
    freqs, times, Sxx = stft(data, fs=samplerate, nperseg=nperseg, noverlap=noverlap)
    return Spectrogram(freqs=freqs, times=times, Sxx=Sxx)
