"""Closed-form Doppler curve fitting per detected ridge.

This module implements its own closed-form forward model of a
constant-velocity straight-line flyby. It is deliberately independent of
`physics.propagation.solve_emission_time` (which is numerical/iterative):
fitting a ridge against this closed form, when the test audio that
produced the ridge was synthesized via the numerical solver, gives a
cross-check between two separately-implemented formulations rather than
testing a model against itself.

Parametrized directly by `(f0, v, d_min, t_heard_ref)`, where
`t_heard_ref` is the heard time of the `f = f0` crossing (where radial
velocity v_r = 0, i.e. the moment of closest *emission-time* approach) --
never the ridge's start, peak, midpoint, or steepest-slope point, since
`t_heard_ref` is exactly the instant the delayed-CPA computation in
grid/coarse.py predicts.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from ..physics.constants import SPEED_OF_SOUND_MPS
from ..physics.doppler import doppler_shift
from ..types import DopplerEvent
from .events import Ridge


def _modeled_frequency(
    t_heard: np.ndarray, f0: float, v: float, d_min: float, t_heard_ref: float, c: float
) -> np.ndarray:
    """Closed-form Doppler curve for a constant-velocity straight-line flyby.

    Derived the same way as the oracle in test_physics_propagation.py: let
    T = t_heard - t_cpa_emission, where t_cpa_emission = t_heard_ref - d_min/c
    is the emission time of closest approach. Solving
    tau + sqrt(v^2 tau^2 + d_min^2)/c = T for tau (minus root, the
    physically valid one) gives the emission-time offset from closest
    approach, from which radial velocity and then frequency follow.
    """
    big_t = t_heard - t_heard_ref + d_min / c
    c2, v2 = c * c, v * v
    discriminant = np.maximum(c2 * v2 * big_t**2 + d_min**2 * (c2 - v2), 0.0)
    tau = (c2 * big_t - np.sqrt(discriminant)) / (c2 - v2)
    distance = np.sqrt(v2 * tau**2 + d_min**2)
    v_r = -v2 * tau / distance
    return doppler_shift(f0, v_r, c)


def fit_doppler_event(ridge: Ridge, event_id: int, c: float = SPEED_OF_SOUND_MPS) -> DopplerEvent:
    times, freqs = ridge.times, ridge.freqs

    f_max, f_min = float(freqs.max()), float(freqs.min())
    f0_guess = 2.0 * f_max * f_min / (f_max + f_min)
    v_guess = min(c * (f_max - f_min) / (f_max + f_min), 0.5 * c)
    t_ref_guess = float(times[np.argmin(np.abs(freqs - f0_guess))])
    slope = float(np.max(np.abs(np.gradient(freqs, times))))
    d_min_guess = max(f0_guess * v_guess**2 / (c * max(slope, 1.0)), 10.0)

    p0 = [f0_guess, max(v_guess, 1.0), d_min_guess, t_ref_guess]
    lower = [1.0, 1.0, 1.0, times.min() - 60.0]
    upper = [20000.0, 0.9 * c, 50000.0, times.max() + 60.0]

    popt, pcov = curve_fit(
        lambda t, f0, v, d_min, t_ref: _modeled_frequency(t, f0, v, d_min, t_ref, c),
        times,
        freqs,
        p0=p0,
        bounds=(lower, upper),
        maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))
    f0_fit, v_fit, d_min_fit, t_ref_fit = (float(x) for x in popt)
    f0_err, v_err, d_min_err, t_ref_err = (float(x) for x in perr)

    return DopplerEvent(
        event_id=event_id,
        t_heard_start=float(times.min()),
        t_heard_end=float(times.max()),
        t_heard_ref=t_ref_fit,
        t_heard_ref_err=t_ref_err,
        f0_estimate=f0_fit,
        f0_estimate_err=f0_err,
        v_estimate=v_fit,
        v_estimate_err=v_err,
        d_min_estimate=d_min_fit,
        d_min_estimate_err=d_min_err,
        curve_times=times,
        curve_freqs=freqs,
    )
