"""Sanity tests for the curved multi-phase flight generator used by the
validation harness (tests/fixtures/simulate_airspace.py) -- not testing
pipeline accuracy, just that the generated flights are physically sane:
a real climb/cruise/descent altitude profile, bounded speed, exact
start/end coordinates, and strictly increasing timestamps (required by
FlightTrack).
"""

from __future__ import annotations

import numpy as np

from .fixtures.airports import WASHINGTON_AIRPORTS
from .fixtures.simulate_airspace import epoch, generate_flight, generate_world, sample_propeller_profile

import datetime as dt

T0 = epoch(dt.date(2026, 6, 15))


def test_generated_flight_has_climb_cruise_descend_profile():
    rng = np.random.default_rng(42)
    origin, destination = WASHINGTON_AIRPORTS[0], WASHINGTON_AIRPORTS[4]  # SEA -> OLM, a longer hop
    profile = sample_propeller_profile(rng, origin, destination)
    track = generate_flight("TESTCLIMB", origin, destination, T0, profile)

    assert track.alt_m[0] == origin.elev_m
    assert track.alt_m[-1] == destination.elev_m
    assert track.alt_m.max() > origin.elev_m + 100.0, "should climb well above the origin airport"
    assert track.alt_m.max() > destination.elev_m + 100.0, "should climb well above the destination airport"

    # A real cruise plateau: altitude should hold near its max for a
    # meaningful stretch, not just touch the peak and immediately descend.
    near_peak = np.abs(track.alt_m - track.alt_m.max()) < 50.0
    assert near_peak.sum() >= 3, "expected a sustained cruise plateau near the peak altitude"


def test_generated_flight_starts_and_ends_at_airports():
    rng = np.random.default_rng(7)
    origin, destination = WASHINGTON_AIRPORTS[1], WASHINGTON_AIRPORTS[2]
    profile = sample_propeller_profile(rng, origin, destination)
    track = generate_flight("TESTENDPOINTS", origin, destination, T0, profile)

    assert track.lat[0] == origin.lat and track.lon[0] == origin.lon

    # Dead reckoning stops as soon as it's within `landing_threshold_m` of
    # the destination -- it doesn't teleport the final sample exactly onto
    # the airport's published coordinates, since that would imply an
    # arbitrarily fast final ground speed between the last two samples.
    dlat_m = (track.lat[-1] - destination.lat) * 111_320.0
    dlon_m = (track.lon[-1] - destination.lon) * 111_320.0 * np.cos(np.radians(destination.lat))
    assert np.hypot(dlat_m, dlon_m) <= 400.0
    assert np.all(np.diff(track.times) > 0), "FlightTrack requires strictly increasing timestamps"


def test_generated_flight_speed_stays_within_sampled_profile_bounds():
    rng = np.random.default_rng(99)
    origin, destination = WASHINGTON_AIRPORTS[3], WASHINGTON_AIRPORTS[6]
    profile = sample_propeller_profile(rng, origin, destination)
    track = generate_flight("TESTSPEED", origin, destination, T0, profile)

    dt_s = np.diff(track.times)
    dlat_m = np.diff(track.lat) * 111_320.0
    dlon_m = np.diff(track.lon) * 111_320.0 * np.cos(np.radians(track.lat[:-1]))
    ground_speed_mps = np.hypot(dlat_m, dlon_m) / dt_s

    # Generous margin: the harness's acceleration/deceleration ramps are a
    # heuristic, not an exact bound, so allow some overshoot rather than
    # pinning to profile.cruise_speed_mps exactly.
    assert ground_speed_mps.max() <= profile.cruise_speed_mps * 1.2


def test_generate_world_returns_matching_tracks_and_f0_map():
    rng = np.random.default_rng(3)
    tracks, f0_by_icao24 = generate_world(rng, WASHINGTON_AIRPORTS, T0, T0 + 86400.0, n_flights=5)

    assert len(tracks) == 5
    assert set(f0_by_icao24) == {t.icao24 for t in tracks}
    for f0 in f0_by_icao24.values():
        assert 70.0 <= f0 <= 220.0
