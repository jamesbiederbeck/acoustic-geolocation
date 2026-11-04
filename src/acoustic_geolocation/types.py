"""Shared dataclasses passed between pipeline stages.

All dataclasses here use `eq=False`: several carry ndarray fields, and the
auto-generated `__eq__`/`__hash__` from `frozen=True` either raises
("ambiguous truth value") or crashes (ndarrays are unhashable) the moment
anything tries to compare, dedupe, or hash one of these.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class DopplerEvent:
    """A single Doppler curve segmented out of a recording's spectrogram."""

    event_id: int
    t_heard_start: float
    t_heard_end: float
    t_heard_ref: float  # heard time of the f = f0 crossing, where v_r = 0 -- the canonical event time
    t_heard_ref_err: float
    f0_estimate: float
    f0_estimate_err: float
    v_estimate: float
    v_estimate_err: float
    d_min_estimate: float
    d_min_estimate_err: float
    curve_times: np.ndarray
    curve_freqs: np.ndarray
