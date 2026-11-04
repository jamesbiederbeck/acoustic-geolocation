"""Validate solve_emission_time against an independently-derived closed-form
oracle for a constant-velocity straight-line flyby, and prove the README's
one-shot pseudocode formula is measurably wrong away from closest approach.

The oracle is derived and implemented only here -- it must never import
from acoustic_geolocation.physics.propagation, or the test would no longer
be independent of the code it's checking.

Scenario (flat, non-geodetic coordinates -- solve_emission_time only cares
about Euclidean ECEF-shaped arrays, so this is a faithful stand-in):
observer fixed at (0, d_min, 0); source moving at constant speed v along
the x-axis, so its closest (non-delayed) approach to the observer, at
emission time `t_cpa_emission`, is the point (0, 0, 0), distance d_min.
"""

import numpy as np

from acoustic_geolocation.physics.propagation import solve_emission_time

C = 343.0
V = 100.0
D_MIN = 2000.0
T_CPA_EMISSION = 1_700_000_000.0  # epoch-scale, per advisor guidance on float precision
OBSERVER = np.array([0.0, D_MIN, 0.0])


def position_fn(t_e):
    t_e = np.asarray(t_e, dtype=float)
    tau = t_e - T_CPA_EMISSION
    x = V * tau
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=-1)


def velocity_fn(t_e):
    t_e = np.asarray(t_e, dtype=float)
    shape = t_e.shape + (3,)
    vel = np.zeros(shape)
    vel[..., 0] = V
    return vel


def _slant_distance_at(t_e):
    pos = position_fn(t_e)
    return np.linalg.norm(OBSERVER - pos, axis=-1)


def oracle_emission_time(t_heard, t_cpa_emission=T_CPA_EMISSION, v=V, d_min=D_MIN, c=C):
    """Independent closed-form solution of tau + sqrt(v^2 tau^2 + d_min^2)/c = T,
    where tau = t_e - t_cpa_emission and T = t_heard - t_cpa_emission.

    Derivation: squaring c(T - tau) = sqrt(v^2 tau^2 + d_min^2) and
    collecting terms in tau gives the quadratic
        (c^2 - v^2) tau^2 - 2 c^2 T tau + (c^2 T^2 - d_min^2) = 0
    whose physically-correct root (the one satisfying T - tau >= 0, i.e.
    sound travel time is non-negative) is the minus branch:
        tau = [c^2 T - sqrt(c^2 v^2 T^2 + d_min^2 (c^2 - v^2))] / (c^2 - v^2)
    """
    t_heard = np.asarray(t_heard, dtype=float)
    big_t = t_heard - t_cpa_emission
    c2, v2 = c * c, v * v
    discriminant = c2 * v2 * big_t**2 + d_min**2 * (c2 - v2)
    tau = (c2 * big_t - np.sqrt(discriminant)) / (c2 - v2)
    return t_cpa_emission + tau


def _true_heard_time(tau_true):
    t_e_true = T_CPA_EMISSION + tau_true
    d = _slant_distance_at(t_e_true)
    return t_e_true + d / C, t_e_true


def test_oracle_agrees_with_direct_forward_computation_at_cpa():
    # Sanity-check the oracle itself: at tau=0, position is exactly d_min away.
    t_heard, t_e_true = _true_heard_time(0.0)
    t_e_oracle = oracle_emission_time(t_heard)
    np.testing.assert_allclose(t_e_oracle, t_e_true, atol=1e-6)


def test_solver_matches_oracle_during_approach():
    t_heard, t_e_true = _true_heard_time(-200.0)
    t_e_est = solve_emission_time(t_heard, OBSERVER, position_fn, velocity_fn, c=C)
    t_e_oracle = oracle_emission_time(t_heard)
    np.testing.assert_allclose(t_e_est, t_e_true, atol=1e-6)
    np.testing.assert_allclose(t_e_est, t_e_oracle, atol=1e-6)


def test_solver_matches_oracle_at_cpa():
    t_heard, t_e_true = _true_heard_time(0.0)
    t_e_est = solve_emission_time(t_heard, OBSERVER, position_fn, velocity_fn, c=C)
    t_e_oracle = oracle_emission_time(t_heard)
    np.testing.assert_allclose(t_e_est, t_e_true, atol=1e-6)
    np.testing.assert_allclose(t_e_est, t_e_oracle, atol=1e-6)


def test_solver_matches_oracle_during_recede():
    t_heard, t_e_true = _true_heard_time(200.0)
    t_e_est = solve_emission_time(t_heard, OBSERVER, position_fn, velocity_fn, c=C)
    t_e_oracle = oracle_emission_time(t_heard)
    np.testing.assert_allclose(t_e_est, t_e_true, atol=1e-6)
    np.testing.assert_allclose(t_e_est, t_e_oracle, atol=1e-6)


def test_solver_vectorizes_over_batch_of_heard_times():
    taus = np.array([-200.0, -50.0, 0.0, 50.0, 200.0])
    t_heards = []
    t_e_trues = []
    for tau in taus:
        t_heard, t_e_true = _true_heard_time(tau)
        t_heards.append(t_heard)
        t_e_trues.append(t_e_true)
    t_heards = np.array(t_heards)
    t_e_trues = np.array(t_e_trues)

    t_e_est = solve_emission_time(t_heards, OBSERVER, position_fn, velocity_fn, c=C)
    np.testing.assert_allclose(t_e_est, t_e_trues, atol=1e-6)


def test_one_shot_seed_is_measurably_wrong_away_from_cpa():
    """The README's pseudocode `emission_time = t - distance(position_at(t))/c`
    is only the first iterate of the solve, not a converged answer. At a
    heard time well into the approach phase (large relative velocity /
    delay), it must differ from the true emission time by much more than
    the solver's convergence tolerance.
    """
    t_heard, t_e_true = _true_heard_time(-200.0)
    one_shot = t_heard - _slant_distance_at(t_heard) / C

    converged = solve_emission_time(t_heard, OBSERVER, position_fn, velocity_fn, c=C)

    assert abs(one_shot - t_e_true) > 1e-3
    assert abs(converged - t_e_true) < 1e-6


def test_out_of_range_times_return_nan():
    """position_fn/velocity_fn returning NaN for out-of-range times must
    propagate to NaN in the result, not raise or silently return garbage.
    """

    def bounded_position_fn(t_e):
        t_e = np.asarray(t_e, dtype=float)
        pos = position_fn(t_e)
        out_of_range = np.abs(t_e - T_CPA_EMISSION) > 1000.0
        pos = pos.copy()
        pos[out_of_range] = np.nan
        return pos

    def bounded_velocity_fn(t_e):
        t_e = np.asarray(t_e, dtype=float)
        vel = velocity_fn(t_e)
        out_of_range = np.abs(t_e - T_CPA_EMISSION) > 1000.0
        vel = vel.copy()
        vel[out_of_range] = np.nan
        return vel

    far_future_heard_time = T_CPA_EMISSION + 1e6
    result = solve_emission_time(far_future_heard_time, OBSERVER, bounded_position_fn, bounded_velocity_fn, c=C)
    assert np.isnan(result)
