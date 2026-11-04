"""Tests for FlightTrack's position/velocity interpolation: exact
reproduction of the underlying straight-line-in-ECEF motion, bounded
linear extrapolation past each end, and the solve_emission_time
integration this was specifically designed for -- commit 2's chosen seed,
`t_heard - d(position_fn(t_heard))/c`, breaks for sound heard after a
track's last sample unless position_fn/velocity_fn extrapolate a bit past
the edges first.
"""

import numpy as np

from acoustic_geolocation.physics.propagation import solve_emission_time
from acoustic_geolocation.types import FlightTrack, TrackSample

from .fixtures.synth_tracks import make_straight_track

T_ANCHOR = 1_700_000_000.0


def _flight_track_from_synth(synth_track, extrapolation_margin_s: float = 60.0) -> FlightTrack:
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
        extrapolation_margin_s=extrapolation_margin_s,
    )


def _make_track(t_start, t_end, extrapolation_margin_s=60.0, dt_s=1.0):
    synth = make_straight_track(
        "AAA111",
        47.0,
        -122.0,
        1500.0,
        heading_deg=90.0,
        speed_mps=90.0,
        t_anchor=T_ANCHOR,
        t_start=t_start,
        t_end=t_end,
        is_propeller=True,
        dt_s=dt_s,
    )
    return _flight_track_from_synth(synth, extrapolation_margin_s=extrapolation_margin_s)


def test_velocity_ecef_is_constant_for_straight_line():
    track = _make_track(T_ANCHOR - 300, T_ANCHOR + 300)

    v1 = track.velocity_ecef(T_ANCHOR - 50.0)
    v2 = track.velocity_ecef(T_ANCHOR + 50.0)
    np.testing.assert_allclose(v1, v2, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(v1), 90.0, atol=1e-6)


def test_position_ecef_matches_velocity_times_elapsed_in_range():
    track = _make_track(T_ANCHOR - 300, T_ANCHOR + 300)
    p0 = track.position_ecef(T_ANCHOR)
    v0 = track.velocity_ecef(T_ANCHOR)
    p_later = track.position_ecef(T_ANCHOR + 42.0)
    np.testing.assert_allclose(p_later, p0 + v0 * 42.0, atol=1e-6)


def test_vectorized_batch_query():
    track = _make_track(T_ANCHOR - 300, T_ANCHOR + 300)
    ts = np.array([T_ANCHOR - 100.0, T_ANCHOR, T_ANCHOR + 100.0])
    positions = track.position_ecef(ts)
    assert positions.shape == (3, 3)
    velocities = track.velocity_ecef(ts)
    assert velocities.shape == (3, 3)


def test_extrapolates_linearly_past_track_end_within_margin():
    track = _make_track(T_ANCHOR - 10, T_ANCHOR + 10, extrapolation_margin_s=60.0)

    t_query = T_ANCHOR + 10 + 5.0  # 5s past the last sample, within margin
    extrapolated = track.position_ecef(t_query)
    assert not np.any(np.isnan(extrapolated))

    expected = track.position_ecef(T_ANCHOR + 10.0) + track.velocity_ecef(T_ANCHOR + 10.0) * 5.0
    np.testing.assert_allclose(extrapolated, expected, atol=1e-3)


def test_returns_nan_beyond_extrapolation_margin():
    track = _make_track(T_ANCHOR - 10, T_ANCHOR + 10, extrapolation_margin_s=60.0)

    far_beyond = T_ANCHOR + 10 + 1000.0
    assert np.all(np.isnan(track.position_ecef(far_beyond)))
    assert np.all(np.isnan(track.velocity_ecef(far_beyond)))


def test_solve_emission_time_resolves_sound_heard_after_track_end():
    """Regression test for the commit-2 seed bug: sound emitted shortly
    before the track's last sample, from ~2km away, is heard several
    seconds after that last sample. Without bounded extrapolation,
    solve_emission_time's one-shot seed evaluates position_fn at a time
    past the track's end and would die on NaN even though the true
    emission time is well within the track's sampled range.
    """
    track = _make_track(T_ANCHOR - 300, T_ANCHOR + 1.0, extrapolation_margin_s=60.0)

    observer_ecef = track.position_ecef(T_ANCHOR) + np.array([0.0, 2000.0, 0.0])
    t_e_true = T_ANCHOR - 1.0  # 1s before the track's last sample
    d_true = np.linalg.norm(observer_ecef - track.position_ecef(t_e_true))
    t_heard = t_e_true + d_true / 343.0  # lands a few seconds past the track's last sample

    assert t_heard > track.t_end  # confirm this test actually exercises the past-the-end case

    t_e_est = solve_emission_time(t_heard, observer_ecef, track.position_ecef, track.velocity_ecef, c=343.0)
    assert not np.isnan(t_e_est)
    np.testing.assert_allclose(t_e_est, t_e_true, atol=1e-3)
