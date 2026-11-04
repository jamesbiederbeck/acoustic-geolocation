"""Geodetic <-> ECEF conversion checks.

A round-trip test alone can't catch a lat/lon argument swap (pyproj's
`always_xy=True` expects (lon, lat, h); swapping on both sides of a
conversion still round-trips cleanly). The anchor-point tests below pin
down absolute ECEF coordinates for known lat/lon/alt so a swap is caught.
"""

import numpy as np

from acoustic_geolocation.physics.constants import (
    WGS84_SEMI_MAJOR_AXIS_M,
    WGS84_SEMI_MINOR_AXIS_M,
)
from acoustic_geolocation.physics.geodesy import ecef_to_geodetic, geodetic_to_ecef


def test_equator_prime_meridian_anchor():
    ecef = geodetic_to_ecef(lat_deg=0.0, lon_deg=0.0, alt_m=0.0)
    np.testing.assert_allclose(ecef, [WGS84_SEMI_MAJOR_AXIS_M, 0.0, 0.0], atol=1e-3)


def test_equator_90east_anchor():
    ecef = geodetic_to_ecef(lat_deg=0.0, lon_deg=90.0, alt_m=0.0)
    np.testing.assert_allclose(ecef, [0.0, WGS84_SEMI_MAJOR_AXIS_M, 0.0], atol=1e-3)


def test_north_pole_anchor():
    ecef = geodetic_to_ecef(lat_deg=90.0, lon_deg=0.0, alt_m=0.0)
    np.testing.assert_allclose(ecef, [0.0, 0.0, WGS84_SEMI_MINOR_AXIS_M], atol=1e-2)


def test_round_trip_arbitrary_point():
    lat, lon, alt = 47.6062, -122.3321, 123.4
    ecef = geodetic_to_ecef(lat, lon, alt)
    lat2, lon2, alt2 = ecef_to_geodetic(ecef)
    np.testing.assert_allclose(lat2, lat, atol=1e-9)
    np.testing.assert_allclose(lon2, lon, atol=1e-9)
    np.testing.assert_allclose(alt2, alt, atol=1e-3)


def test_vectorized_batch():
    lat = np.array([0.0, 45.0, -33.9])
    lon = np.array([0.0, -120.0, 151.2])
    alt = np.array([0.0, 1000.0, 50.0])
    ecef = geodetic_to_ecef(lat, lon, alt)
    assert ecef.shape == (3, 3)
    lat2, lon2, alt2 = ecef_to_geodetic(ecef)
    np.testing.assert_allclose(lat2, lat, atol=1e-9)
    np.testing.assert_allclose(lon2, lon, atol=1e-9)
    np.testing.assert_allclose(alt2, alt, atol=1e-3)
