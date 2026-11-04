"""Synthetic ADS-B flight tracks for tests.

Built as straight lines in ECEF, not lat/lon/alt: a constant-heading,
constant-altitude lat/lon path is not straight in ECEF over tens of
kilometers (it bows by several meters), which would make "linear ECEF
interpolation reproduces the track exactly" a subtly wrong thing to
assert against lat/lon-derived ground truth. These tracks are straight in
ECEF by construction, converted to lat/lon/alt only when materializing a
CSV dump -- mirroring what a real ADS-B feed provides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from acoustic_geolocation.physics.geodesy import ecef_to_geodetic, geodetic_to_ecef


def _local_enu_basis(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray]:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    return east, north


@dataclass
class SynthTrack:
    icao24: str
    callsign: str
    aircraft_type: str
    is_propeller: bool
    times: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt_m: np.ndarray


def make_straight_track(
    icao24: str,
    anchor_lat: float,
    anchor_lon: float,
    anchor_alt_m: float,
    heading_deg: float,
    speed_mps: float,
    t_anchor: float,
    t_start: float,
    t_end: float,
    is_propeller: bool,
    aircraft_type: str = "GENERIC",
    dt_s: float = 1.0,
) -> SynthTrack:
    """A constant-velocity straight line in ECEF passing through
    `(anchor_lat, anchor_lon, anchor_alt_m)` at `t_anchor`, heading
    `heading_deg` (measured in the anchor's local ENU frame, then held
    fixed as a constant ECEF velocity vector for the whole track).
    """
    anchor_ecef = geodetic_to_ecef(anchor_lat, anchor_lon, anchor_alt_m)
    east, north = _local_enu_basis(anchor_lat, anchor_lon)
    heading = np.radians(heading_deg)
    direction = np.sin(heading) * east + np.cos(heading) * north
    velocity_ecef = speed_mps * direction

    times = np.arange(t_start, t_end + dt_s / 2.0, dt_s)
    positions = anchor_ecef + velocity_ecef[None, :] * (times - t_anchor)[:, None]
    lat, lon, alt_m = ecef_to_geodetic(positions)

    return SynthTrack(
        icao24=icao24,
        callsign=f"CS{icao24}",
        aircraft_type=aircraft_type,
        is_propeller=is_propeller,
        times=times,
        lat=np.asarray(lat),
        lon=np.asarray(lon),
        alt_m=np.asarray(alt_m),
    )


def tracks_to_dataframe(tracks: list[SynthTrack]) -> pd.DataFrame:
    rows = []
    for track in tracks:
        for i in range(len(track.times)):
            rows.append(
                {
                    "icao24": track.icao24,
                    "callsign": track.callsign,
                    "aircraft_type": track.aircraft_type,
                    "timestamp": float(track.times[i]),
                    "lat": float(track.lat[i]),
                    "lon": float(track.lon[i]),
                    "alt_m": float(track.alt_m[i]),
                    "is_propeller": track.is_propeller,
                }
            )
    return pd.DataFrame(rows)


def write_adsb_csv(path, tracks: list[SynthTrack]) -> None:
    tracks_to_dataframe(tracks).to_csv(path, index=False)
