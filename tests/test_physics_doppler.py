"""Independent hand-computed checks for radial_velocity and doppler_shift."""

import numpy as np

from acoustic_geolocation.physics.doppler import doppler_shift, radial_velocity


def test_doppler_shift_approaching_value():
    f = doppler_shift(100.0, 100.0, c=343.0)
    np.testing.assert_allclose(f, 34300.0 / 243.0, rtol=1e-9)
    assert f > 100.0


def test_doppler_shift_receding_value():
    f = doppler_shift(100.0, -100.0, c=343.0)
    np.testing.assert_allclose(f, 34300.0 / 443.0, rtol=1e-9)
    assert f < 100.0


def test_doppler_shift_perpendicular_is_unshifted():
    f = doppler_shift(100.0, 0.0, c=343.0)
    np.testing.assert_allclose(f, 100.0, rtol=1e-12)


def test_radial_velocity_approaching_is_positive():
    # Source at origin moving in +x, observer at (10, 0, 0): closing distance.
    v_r = radial_velocity(
        velocity_ecef=np.array([5.0, 0.0, 0.0]),
        position_ecef=np.array([0.0, 0.0, 0.0]),
        observer_ecef=np.array([10.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(v_r, 5.0)


def test_radial_velocity_receding_is_negative():
    v_r = radial_velocity(
        velocity_ecef=np.array([-5.0, 0.0, 0.0]),
        position_ecef=np.array([0.0, 0.0, 0.0]),
        observer_ecef=np.array([10.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(v_r, -5.0)


def test_radial_velocity_perpendicular_is_zero():
    # Source at (0, 0, 0) moving in +y, observer at (10, 0, 0): motion is
    # entirely tangential, no radial component.
    v_r = radial_velocity(
        velocity_ecef=np.array([0.0, 5.0, 0.0]),
        position_ecef=np.array([0.0, 0.0, 0.0]),
        observer_ecef=np.array([10.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(v_r, 0.0, atol=1e-12)


def test_radial_velocity_broadcasts_over_batch():
    velocity = np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    position = np.zeros((3, 3))
    observer = np.array([10.0, 0.0, 0.0])
    v_r = radial_velocity(velocity, position, observer)
    np.testing.assert_allclose(v_r, [5.0, -5.0, 0.0], atol=1e-12)
