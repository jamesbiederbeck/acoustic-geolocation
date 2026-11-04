"""Tests for the coarse grid filter: recovering the true observer point
from two non-parallel flights' predicted delayed-CPA timing/distance, and
rejecting candidates when a third aircraft (a distractor jet, not one of
the two matched flights) would have been audible between the two events.

The two matched tracks' ground truth is read back via delayed_cpa() at
the truth point itself, rather than independently hand-derived -- this
test is about coarse_filter's own logic (pairing, tolerances, silence
check), not re-validating the underlying physics (covered by
test_physics_propagation.py and test_flight_track.py).
"""

import numpy as np

from acoustic_geolocation.adsb.interface import BBox
from acoustic_geolocation.grid.coarse import coarse_filter, delayed_cpa, generate_grid
from acoustic_geolocation.grid.coarse import _min_distance_in_heard_window
from acoustic_geolocation.physics.geodesy import FlatDem, geodetic_to_ecef
from acoustic_geolocation.types import DopplerEvent, GridPoint

from .fixtures.synth_tracks import make_straight_track

T_ANCHOR = 1_700_000_000.0
TRUTH_LAT, TRUTH_LON, TRUTH_ELEV_M = 47.05, -122.05, 50.0
BBOX = BBox(lat_min=46.95, lon_min=-122.15, lat_max=47.15, lon_max=-121.95)
RESOLUTION_M = 500.0


def _truth_point() -> GridPoint:
    ecef = geodetic_to_ecef(TRUTH_LAT, TRUTH_LON, TRUTH_ELEV_M)
    return GridPoint(lat=TRUTH_LAT, lon=TRUTH_LON, elevation_m=TRUTH_ELEV_M, ecef=ecef)


def _flight_track_from_synth(synth_track):
    from acoustic_geolocation.types import FlightTrack, TrackSample

    samples = tuple(
        TrackSample(t=float(t), lat=float(lat), lon=float(lon), alt_m=float(alt))
        for t, lat, lon, alt in zip(synth_track.times, synth_track.lat, synth_track.lon, synth_track.alt_m)
    )
    return FlightTrack(
        icao24=synth_track.icao24,
        callsign=synth_track.callsign,
        aircraft_type=synth_track.aircraft_type,
        is_propeller=synth_track.is_propeller,
        samples=samples,
    )


def _make_track1():
    # Heading due east, offset ~1.5km north of truth -- closest approach
    # (in longitude) lands near t_anchor, non-parallel to track2 below.
    synth = make_straight_track(
        "PROP1", 47.064, -122.05, 1500.0, heading_deg=90.0, speed_mps=90.0,
        t_anchor=T_ANCHOR, t_start=T_ANCHOR - 300, t_end=T_ANCHOR + 300, is_propeller=True,
    )
    return _flight_track_from_synth(synth)


def _make_track2():
    # Heading due north, offset ~1.2km east of truth, closest approach
    # ~150s after track1's -- non-parallel, time-separated.
    synth = make_straight_track(
        "PROP2", 47.05, -122.034, 1800.0, heading_deg=0.0, speed_mps=100.0,
        t_anchor=T_ANCHOR + 150.0, t_start=T_ANCHOR - 150, t_end=T_ANCHOR + 450, is_propeller=True,
    )
    return _flight_track_from_synth(synth)


def _make_distractor_jet():
    # Passes within ~110m horizontally of truth, roughly midway through
    # the window between track1's and track2's events -- should be
    # audible there (altitude kept modest, not realistic for a jet, so
    # slant distance stays under the default audibility radius).
    synth = make_straight_track(
        "JET999", 47.051, -122.05, 2000.0, heading_deg=270.0, speed_mps=230.0,
        t_anchor=T_ANCHOR + 75.0, t_start=T_ANCHOR - 50, t_end=T_ANCHOR + 200, is_propeller=False,
    )
    return _flight_track_from_synth(synth)


def _make_post_window_distractor_jet():
    """Its own closest approach lands ~6s *after* track2's event (outside
    the two-event window), but it's fast and close enough to already be
    well within the audibility radius for the tail end of the window --
    the specific case _min_distance_in_heard_window exists to catch,
    which a CPA-time-only check would wrongly call inaudible.
    """
    synth = make_straight_track(
        "JET888", 47.0505, -122.05, 800.0, heading_deg=180.0, speed_mps=250.0,
        t_anchor=T_ANCHOR + 160.0, t_start=T_ANCHOR - 50, t_end=T_ANCHOR + 250, is_propeller=False,
    )
    return _flight_track_from_synth(synth)


def _make_events(track1, track2, t0_true=0.0):
    """Build DopplerEvents whose t_heard_* fields are *recording-relative*
    (absolute heard time minus `t0_true`), matching what the real audio
    stage produces -- so a test using `t0_true != 0` can actually catch a
    relative/absolute time mixup in coarse_filter's t0_hypothesis, which
    an all-zero t0_true would hide.
    """
    truth = _truth_point()
    t_heard_1, d_min_1 = delayed_cpa(truth, track1)
    t_heard_2, d_min_2 = delayed_cpa(truth, track2)
    t_heard_1 -= t0_true
    t_heard_2 -= t0_true

    placeholder = np.array([0.0])
    event_1 = DopplerEvent(
        event_id=0, t_heard_start=t_heard_1 - 10, t_heard_end=t_heard_1 + 10,
        t_heard_ref=t_heard_1, t_heard_ref_err=0.05,
        f0_estimate=400.0, f0_estimate_err=1.0,
        v_estimate=90.0, v_estimate_err=5.0,
        d_min_estimate=d_min_1, d_min_estimate_err=5.0,
        curve_times=placeholder, curve_freqs=placeholder,
    )
    event_2 = DopplerEvent(
        event_id=1, t_heard_start=t_heard_2 - 10, t_heard_end=t_heard_2 + 10,
        t_heard_ref=t_heard_2, t_heard_ref_err=0.05,
        f0_estimate=500.0, f0_estimate_err=1.0,
        v_estimate=100.0, v_estimate_err=5.0,
        d_min_estimate=d_min_2, d_min_estimate_err=5.0,
        curve_times=placeholder, curve_freqs=placeholder,
    )
    return event_1, event_2


def _nearest_grid_point(points):
    def planar_dist_deg(p):
        return (p.lat - TRUTH_LAT) ** 2 + (p.lon - TRUTH_LON) ** 2

    return min(points, key=planar_dist_deg)


def _distance_m(lat1, lon1, lat2, lon2):
    lat_mid = np.radians(0.5 * (lat1 + lat2))
    dlat_m = (lat1 - lat2) * _METERS_PER_DEGREE_LAT
    dlon_m = (lon1 - lon2) * _METERS_PER_DEGREE_LAT * np.cos(lat_mid)
    return float(np.hypot(dlat_m, dlon_m))


_METERS_PER_DEGREE_LAT = 111_320.0
T0_TRUE = T_ANCHOR - 30.0

# Two non-parallel tracks can have a near-mirror-image candidate location
# where both d_min values still roughly match and the timing offset shift
# stays inside the coarse margin -- a known coarse-stage limitation, not a
# bug, so this bounds *all* survivors rather than asserting there is only
# one. Pinning it down precisely is left to the fine-grid scoring stage.
_SURVIVOR_SANITY_RADIUS_M = 6000.0


def test_coarse_filter_recovers_true_point_flight_pairing_and_t0():
    track1, track2 = _make_track1(), _make_track2()
    event_1, event_2 = _make_events(track1, track2, t0_true=T0_TRUE)

    points = generate_grid(BBOX, RESOLUTION_M, FlatDem(TRUTH_ELEV_M))
    nearest = _nearest_grid_point(points)

    candidates = coarse_filter(
        points, (event_1, event_2), match_tracks=[track1, track2], silence_tracks=[track1, track2],
        resolution_m=RESOLUTION_M,
    )

    assert len(candidates) > 0
    for candidate in candidates:
        assert _distance_m(candidate.point.lat, candidate.point.lon, TRUTH_LAT, TRUTH_LON) < _SURVIVOR_SANITY_RADIUS_M

    matches = [c for c in candidates if c.point is nearest and c.flight_1 is track1 and c.flight_2 is track2]
    assert len(matches) == 1, "the grid cell nearest the truth point, with the correct flight pairing, must survive"
    assert abs(matches[0].t0_hypothesis - T0_TRUE) < 10.0, "t0_hypothesis must recover T0 (relative/absolute mixup)"

    swapped = [c for c in candidates if c.point is nearest and c.flight_1 is track2 and c.flight_2 is track1]
    assert len(swapped) == 0, "the wrong flight<->event pairing must never survive at the true point"


def test_coarse_filter_rejects_when_distractor_audible_between_events():
    track1, track2 = _make_track1(), _make_track2()
    jet = _make_distractor_jet()
    event_1, event_2 = _make_events(track1, track2)

    points = generate_grid(BBOX, RESOLUTION_M, FlatDem(TRUTH_ELEV_M))
    nearest = _nearest_grid_point(points)

    candidates = coarse_filter(
        points, (event_1, event_2), match_tracks=[track1, track2], silence_tracks=[track1, track2, jet],
        resolution_m=RESOLUTION_M,
    )

    matches = [c for c in candidates if c.point is nearest and c.flight_1 is track1 and c.flight_2 is track2]
    assert len(matches) == 0, "the jet's presence between the two events must suppress this candidate"


def test_silence_filter_catches_distractor_whose_cpa_lands_just_after_the_window():
    """Regression test for the window-aware silence filter: a
    CPA-time-only check would see this jet's own closest approach land
    ~6s after the window closes and wrongly call it inaudible, even
    though it's already well within the audibility radius by the time
    the window ends.
    """
    track1, track2 = _make_track1(), _make_track2()
    jet = _make_post_window_distractor_jet()
    event_1, event_2 = _make_events(track1, track2)
    truth = _truth_point()

    t_heard_1, _ = delayed_cpa(truth, track1)
    t_heard_2, _ = delayed_cpa(truth, track2)
    t_cpa_jet, d_cpa_jet = delayed_cpa(truth, jet)
    assert t_cpa_jet > t_heard_2, "fixture must put the jet's own CPA outside (after) the window"

    assert _min_distance_in_heard_window(truth, jet, (t_heard_1, t_heard_2), 343.0) is not None

    points = generate_grid(BBOX, RESOLUTION_M, FlatDem(TRUTH_ELEV_M))
    nearest = _nearest_grid_point(points)

    # A wider-than-default audibility radius keeps this robust to the
    # ~150m grid-quantization jitter between the truth point (used to
    # size the fixture above) and the nearest grid cell actually tested --
    # the point here is exercising the window-vs-CPA-only distinction,
    # not pinning an exact distance.
    candidates = coarse_filter(
        points, (event_1, event_2), match_tracks=[track1, track2], silence_tracks=[track1, track2, jet],
        resolution_m=RESOLUTION_M, audibility_radius_m=6000.0,
    )

    matches = [c for c in candidates if c.point is nearest and c.flight_1 is track1 and c.flight_2 is track2]
    assert len(matches) == 0
