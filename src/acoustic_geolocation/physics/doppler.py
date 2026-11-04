"""Radial velocity and the Doppler frequency formula.

Sign convention: `radial_velocity` is positive when the source is
approaching the observer (closing distance). This is consistent with
`doppler_shift`'s f = f0 * c / (c - v_r): approaching (v_r > 0) shrinks the
denominator and raises the observed frequency above f0, as expected.
"""

from __future__ import annotations

import numpy as np

from .constants import SPEED_OF_SOUND_MPS


def radial_velocity(velocity_ecef: np.ndarray, position_ecef: np.ndarray, observer_ecef: np.ndarray) -> np.ndarray:
    """Component of `velocity_ecef` (at `position_ecef`) directed toward `observer_ecef`.

    Positive = approaching. Shapes broadcast against each other; the last
    axis must be length 3 (ECEF x, y, z).
    """
    r_vec = np.asarray(observer_ecef, dtype=float) - np.asarray(position_ecef, dtype=float)
    distance = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    direction = r_vec / distance
    return np.sum(np.asarray(velocity_ecef, dtype=float) * direction, axis=-1)


def doppler_shift(f0: np.ndarray, v_r: np.ndarray, c: float = SPEED_OF_SOUND_MPS) -> np.ndarray:
    """Observed frequency f = f0 * c / (c - v_r) for radial velocity v_r (positive = approaching)."""
    return np.asarray(f0, dtype=float) * c / (c - np.asarray(v_r, dtype=float))
