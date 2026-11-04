"""Lat/lon/alt <-> ECEF conversions and the DemSource protocol.

v1 treats barometric altitude (ADS-B), orthometric/MSL elevation (DEM), and
WGS84 ellipsoidal height (what the ECEF transform wants) as equivalent.
This is a known, documented approximation -- no geoid-undulation correction
yet -- not an oversight. See CLAUDE.md.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from pyproj import Transformer

from .constants import WGS84_ECEF_CRS, WGS84_GEODETIC_CRS

_TO_ECEF = Transformer.from_crs(WGS84_GEODETIC_CRS, WGS84_ECEF_CRS, always_xy=True)
_TO_GEODETIC = Transformer.from_crs(WGS84_ECEF_CRS, WGS84_GEODETIC_CRS, always_xy=True)


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
    """lat/lon in degrees, alt in meters (ellipsoidal height) -> ECEF (..., 3) meters."""
    lat_deg = np.asarray(lat_deg, dtype=float)
    lon_deg = np.asarray(lon_deg, dtype=float)
    alt_m = np.asarray(alt_m, dtype=float)
    x, y, z = _TO_ECEF.transform(lon_deg, lat_deg, alt_m)
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECEF (..., 3) meters -> (lat_deg, lon_deg, alt_m)."""
    ecef = np.asarray(ecef, dtype=float)
    lon_deg, lat_deg, alt_m = _TO_GEODETIC.transform(ecef[..., 0], ecef[..., 1], ecef[..., 2])
    return np.asarray(lat_deg), np.asarray(lon_deg), np.asarray(alt_m)


class DemSource(Protocol):
    """Elevation lookup, swappable for a real raster-backed DEM later."""

    def elevation_m(self, lat_deg: float, lon_deg: float) -> float: ...


class FlatDem:
    """Trivial DemSource: constant elevation everywhere. Used for v1 and all tests."""

    def __init__(self, elevation_m: float = 0.0) -> None:
        self._elevation_m = elevation_m

    def elevation_m(self, lat_deg: float, lon_deg: float) -> float:
        return self._elevation_m
