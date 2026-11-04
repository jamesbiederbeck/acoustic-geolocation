"""Sound-propagation delay equation solver: t_heard = t_emitted + d(t_emitted)/c.

Solved via Newton's method rather than plain fixed-point iteration. Plain
fixed-point iteration (t_e <- t_heard - d(t_e)/c) contracts at rate |v_r|/c
per step, which needs ~20 iterations to converge for a fast aircraft at
long range -- too slow to be practical. Newton's method uses the exact
derivative g'(t_e) = 1 - v_r/c (heard-time is strictly monotonic in
emission-time for any subsonic source, since |v_r| < c) and converges in
~4-5 iterations instead.

The seed value `t_heard - d(position_fn(t_heard))/c` is exactly the
README's one-shot pseudocode formula for emission_time -- it is only the
*first* iterate computed here, never the returned answer. See
test_physics_propagation.py for a regression test showing that one-shot
value is measurably wrong away from closest approach.

This module takes plain position/velocity functions rather than a
FlightTrack object, so physics/ has no dependency on the ADS-B/types
modules: `position_fn`/`velocity_fn` map an array of times, shape (...,),
to ECEF arrays, shape (..., 3). Callers are responsible for returning NaN
rows for times outside a track's known range; NaN propagates through the
residual and keeps those elements unconverged, so they come back as NaN.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .constants import SPEED_OF_SOUND_MPS
from .doppler import radial_velocity

PositionFn = Callable[[np.ndarray], np.ndarray]
VelocityFn = Callable[[np.ndarray], np.ndarray]


def _slant_distance(position_ecef: np.ndarray, observer_ecef: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(observer_ecef, dtype=float) - np.asarray(position_ecef, dtype=float), axis=-1)


def solve_emission_time(
    t_heard: np.ndarray,
    observer_ecef: np.ndarray,
    position_fn: PositionFn,
    velocity_fn: VelocityFn,
    c: float = SPEED_OF_SOUND_MPS,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> np.ndarray:
    """Solve `t_e + d(t_e)/c = t_heard` for `t_e`, vectorized over `t_heard`.

    Returns an array shaped like `t_heard`. Elements that fail to converge
    within `max_iter` -- including when `position_fn`/`velocity_fn` return
    NaN for out-of-range times, or the implied source is supersonic toward
    the observer (1 - v_r/c <= 0, where Newton's linearization breaks down)
    -- come back as NaN rather than raising, since a grid search is
    expected to probe many invalid candidates.
    """
    t_heard = np.asarray(t_heard, dtype=float)
    t_e = t_heard - _slant_distance(position_fn(t_heard), observer_ecef) / c  # one-shot seed, not the answer
    converged = np.zeros(t_heard.shape, dtype=bool)

    for _ in range(max_iter):
        pos = position_fn(t_e)
        vel = velocity_fn(t_e)
        d = _slant_distance(pos, observer_ecef)
        v_r = radial_velocity(vel, pos, observer_ecef)
        residual = t_e + d / c - t_heard
        converged = converged | (np.abs(residual) < tol)
        denom = 1.0 - v_r / c
        safe_denom = np.where(denom > 1e-6, denom, np.nan)
        step = np.where(converged, 0.0, residual / safe_denom)
        t_e = t_e - step
        if np.all(converged):
            break

    return np.where(converged, t_e, np.nan)
