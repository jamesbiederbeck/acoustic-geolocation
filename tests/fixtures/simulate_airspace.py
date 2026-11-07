"""Curved, multi-phase synthetic flight generation for the validation
harness: unlike `synth_tracks.py::make_straight_track` (a single fixed
heading for the whole track, used by the small per-stage/e2e fixtures),
flights generated here climb away from their origin airport, cruise, and
descend into their destination airport, with a genuine curved departure
and arrival track rather than a straight line.

Generated via dead-reckoning time-stepping in the local ENU frame at the
aircraft's *current* position (`synth_tracks.py::local_enu_basis`), not a
closed-form trajectory: at each step the aircraft's heading is the
bearing to the destination plus a perturbation that starts large at
liftoff and eases to zero over the climb phase (curving away from the
runway) and a smaller perturbation that eases in during the final
descent (a curved approach rather than a straight final leg). Recomputing
the bearing every step, rather than holding one fixed heading, is what
makes the path curve at all -- a fixed heading plus a decaying offset
would just be a straight line with a crooked start.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from acoustic_geolocation.physics.geodesy import geodetic_to_ecef

from .airports import Airport
from .synth_tracks import SynthTrack, local_enu_basis

_METERS_PER_DEGREE_LAT = 111_320.0


@dataclass
class AircraftProfile:
    cruise_speed_mps: float
    cruise_altitude_m: float
    climb_rate_mps: float
    descent_rate_mps: float
    departure_turn_deg: float
    descent_turn_deg: float
    f0_hz: float


def sample_propeller_profile(rng: np.random.Generator, origin: Airport, destination: Airport) -> AircraftProfile:
    """Sample a plausible light-propeller-aircraft flight profile, clipped
    to sane bounds. `cruise_altitude_m` is further capped so the climb and
    descent legs can't exceed the route's direct distance -- a short hop
    between nearby airports shouldn't climb to a cruise altitude it has
    no room to reach and descend from again.
    """
    cruise_speed_mps = float(np.clip(rng.normal(55.0, 12.0), 30.0, 80.0))
    climb_rate_mps = float(np.clip(rng.normal(3.5, 1.0), 1.5, 6.0))
    descent_rate_mps = float(np.clip(rng.normal(3.0, 1.0), 1.5, 5.0))

    origin_ecef = geodetic_to_ecef(origin.lat, origin.lon, 0.0)
    dest_ecef = geodetic_to_ecef(destination.lat, destination.lon, 0.0)
    direct_distance_m = float(np.linalg.norm(dest_ecef - origin_ecef))

    # Cap cruise altitude so climbing from origin.elev_m and descending to
    # destination.elev_m, at these climb/descent rates and cruise speed,
    # fits within ~60% of the direct route distance -- leaving room for an
    # actual cruise leg (and slack for the curved, longer-than-direct path
    # actually flown). Derived from: cruise_speed * ((h-origin.elev_m)/climb_rate
    # + (h-destination.elev_m)/descent_rate) <= 0.6 * direct_distance_m, solved for h.
    rate_sum_inv = 1.0 / climb_rate_mps + 1.0 / descent_rate_mps
    max_cruise_altitude_m = (
        0.6 * direct_distance_m / cruise_speed_mps + origin.elev_m / climb_rate_mps + destination.elev_m / descent_rate_mps
    ) / rate_sum_inv

    base_elev = max(origin.elev_m, destination.elev_m)
    cruise_altitude_m = float(np.clip(rng.normal(2500.0, 700.0), 900.0, 4000.0 + base_elev))
    cruise_altitude_m = max(base_elev + 150.0, min(cruise_altitude_m, max(max_cruise_altitude_m, base_elev + 300.0)))

    departure_turn_deg = float(rng.choice([-1.0, 1.0]) * rng.uniform(40.0, 120.0))
    descent_turn_deg = float(rng.choice([-1.0, 1.0]) * rng.uniform(20.0, 60.0))
    f0_hz = float(np.clip(rng.normal(110.0, 25.0), 70.0, 220.0))

    return AircraftProfile(
        cruise_speed_mps=cruise_speed_mps,
        cruise_altitude_m=cruise_altitude_m,
        climb_rate_mps=climb_rate_mps,
        descent_rate_mps=descent_rate_mps,
        departure_turn_deg=departure_turn_deg,
        descent_turn_deg=descent_turn_deg,
        f0_hz=f0_hz,
    )


def _bearing_rad(cur_lat: float, cur_lon: float, dest_ecef: np.ndarray) -> float:
    cur_ecef = geodetic_to_ecef(cur_lat, cur_lon, 0.0)
    east, north = local_enu_basis(cur_lat, cur_lon)
    diff = dest_ecef - cur_ecef
    e, n = float(np.dot(diff, east)), float(np.dot(diff, north))
    return float(np.arctan2(e, n))


def generate_flight(
    icao24: str,
    origin: Airport,
    destination: Airport,
    depart_t: float,
    profile: AircraftProfile,
    dt_s: float = 5.0,
    landing_threshold_m: float = 400.0,
    max_duration_s: float = 6 * 3600.0,
) -> SynthTrack:
    """Dead-reckon a curved climb -> cruise -> descent flight from
    `origin` to `destination`, departing at absolute epoch time
    `depart_t`. Returns a SynthTrack (same shape `make_straight_track`
    returns), stepping at `dt_s` intervals -- no changes needed to
    SynthTrack/write_adsb_csv/tracks_to_dataframe, which are already
    generic over arbitrary times/lat/lon/alt arrays.
    """
    dest_ecef = geodetic_to_ecef(destination.lat, destination.lon, 0.0)

    climb_duration_s = max((profile.cruise_altitude_m - origin.elev_m) / profile.climb_rate_mps, 1.0)
    descent_duration_s = max((profile.cruise_altitude_m - destination.elev_m) / profile.descent_rate_mps, 1.0)
    climb_accel = 0.7 * profile.cruise_speed_mps / climb_duration_s
    descent_decel = 0.7 * profile.cruise_speed_mps / descent_duration_s

    times, lats, lons, alts = [depart_t], [origin.lat], [origin.lon], [origin.elev_m]
    lat, lon, alt = origin.lat, origin.lon, origin.elev_m
    t_elapsed = 0.0
    descending = False
    t_descent_elapsed = 0.0

    while t_elapsed < max_duration_s:
        cur_ecef = geodetic_to_ecef(lat, lon, 0.0)
        remaining_m = float(np.linalg.norm(dest_ecef - cur_ecef))
        if remaining_m <= landing_threshold_m:
            break

        if not descending:
            # Trigger descent once the remaining distance is roughly what's
            # needed to glide down to destination altitude at descent_rate.
            required_descent_m = (alt - destination.elev_m) / profile.descent_rate_mps * profile.cruise_speed_mps
            if remaining_m <= required_descent_m:
                descending = True

        if alt < profile.cruise_altitude_m and not descending:
            speed = min(profile.cruise_speed_mps, 0.3 * profile.cruise_speed_mps + climb_accel * t_elapsed)
            alt = min(profile.cruise_altitude_m, alt + profile.climb_rate_mps * dt_s)
            offset_deg = profile.departure_turn_deg * max(0.0, 1.0 - t_elapsed / climb_duration_s)
        elif descending:
            speed = max(0.3 * profile.cruise_speed_mps, profile.cruise_speed_mps - descent_decel * t_descent_elapsed)
            alt = max(destination.elev_m, alt - profile.descent_rate_mps * dt_s)
            offset_deg = profile.descent_turn_deg * min(1.0, t_descent_elapsed / descent_duration_s)
            t_descent_elapsed += dt_s
        else:
            speed = profile.cruise_speed_mps
            offset_deg = 0.0

        heading = _bearing_rad(lat, lon, dest_ecef) + np.radians(offset_deg)
        step_m = speed * dt_s
        east_disp = np.sin(heading) * step_m
        north_disp = np.cos(heading) * step_m

        lat = lat + north_disp / _METERS_PER_DEGREE_LAT
        meters_per_degree_lon = _METERS_PER_DEGREE_LAT * max(np.cos(np.radians(lat)), 1e-6)
        lon = lon + east_disp / meters_per_degree_lon

        t_elapsed += dt_s
        times.append(depart_t + t_elapsed)
        lats.append(lat)
        lons.append(lon)
        alts.append(alt)

    return SynthTrack(
        icao24=icao24,
        callsign=f"CS{icao24}",
        aircraft_type="GENERIC_PROP",
        is_propeller=True,
        times=np.asarray(times, dtype=float),
        lat=np.asarray(lats, dtype=float),
        lon=np.asarray(lons, dtype=float),
        alt_m=np.asarray(alts, dtype=float),
    )


def generate_world(
    rng: np.random.Generator,
    airports: tuple[Airport, ...],
    t_start: float,
    t_end: float,
    n_flights: int,
    dt_s: float = 5.0,
) -> tuple[list[SynthTrack], dict[str, float]]:
    """Generate `n_flights` random-airport-pair flights departing at
    random times in `[t_start, t_end]`. Returns the tracks plus a
    `icao24 -> f0_hz` map (each flight's sampled engine-tone frequency,
    not representable on a plain ADS-B record) for `simulate_recording.py`
    to use.
    """
    flights = []
    f0_by_icao24: dict[str, float] = {}
    for i in range(n_flights):
        origin, destination = rng.choice(airports, size=2, replace=False)
        depart_t = float(rng.uniform(t_start, t_end))
        profile = sample_propeller_profile(rng, origin, destination)
        icao24 = f"SIM{i:05d}"
        flights.append(generate_flight(icao24, origin, destination, depart_t, profile, dt_s=dt_s))
        f0_by_icao24[icao24] = profile.f0_hz
    return flights, f0_by_icao24


def epoch(date: dt.date, hour: float = 0.0) -> float:
    """Convenience: UTC epoch seconds for `date` at `hour` (fractional hours past midnight)."""
    return dt.datetime.combine(date, dt.time.min, tzinfo=dt.timezone.utc).timestamp() + hour * 3600.0
