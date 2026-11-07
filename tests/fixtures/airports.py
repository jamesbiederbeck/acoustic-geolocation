"""A static list of real public-use Washington-state airports, used as
departure/arrival points by `simulate_airspace.py`. Coordinates and
elevations are approximate (nearest ~10m) -- fine for a synthetic
validation dataset, not survey-grade.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Airport:
    icao: str
    lat: float
    lon: float
    elev_m: float


WASHINGTON_AIRPORTS: tuple[Airport, ...] = (
    Airport("KSEA", 47.4502, -122.3088, 130.0),  # Seattle-Tacoma Intl
    Airport("KBFI", 47.5300, -122.3019, 5.0),  # Boeing Field / King County Intl
    Airport("KPAE", 47.9063, -122.2816, 192.0),  # Paine Field
    Airport("KGEG", 47.6199, -117.5338, 735.0),  # Spokane Intl
    Airport("KOLM", 46.9694, -122.9032, 61.0),  # Olympia Regional
    Airport("KBLI", 48.7928, -122.5375, 59.0),  # Bellingham Intl
    Airport("KYKM", 46.5682, -120.5440, 325.0),  # Yakima Air Terminal
    Airport("KPSC", 46.2647, -119.1189, 124.0),  # Tri-Cities
    Airport("KALW", 46.0949, -118.2878, 366.0),  # Walla Walla Regional
    Airport("KEAT", 47.3987, -120.2073, 372.0),  # Pangborn Memorial (Wenatchee)
    Airport("KFHR", 48.5218, -123.0241, 38.0),  # Friday Harbor
    Airport("KCLM", 48.1201, -123.5000, 89.0),  # William R. Fairchild Intl (Port Angeles)
    Airport("KMWH", 47.2079, -119.3195, 366.0),  # Grant County Intl (Moses Lake)
)
