"""Tests for LocalFileAdsbSource: track grouping, hygiene (sort, dedup,
gap-splitting), and UTC-day padding across date boundaries.
"""

import datetime as dt

import pandas as pd
import numpy as np

from acoustic_geolocation.adsb.interface import BBox
from acoustic_geolocation.adsb.local_file import LocalFileAdsbSource

from .fixtures.synth_tracks import make_straight_track, tracks_to_dataframe, write_adsb_csv

T_ANCHOR = 1_700_000_000.0
BBOX = BBox(lat_min=40.0, lon_min=-130.0, lat_max=55.0, lon_max=-110.0)


def _anchor_date_utc() -> dt.date:
    return dt.datetime.fromtimestamp(T_ANCHOR, tz=dt.timezone.utc).date()


def test_loads_single_track(tmp_path):
    track = make_straight_track(
        "AAA111",
        47.0,
        -122.0,
        1500.0,
        heading_deg=90.0,
        speed_mps=90.0,
        t_anchor=T_ANCHOR,
        t_start=T_ANCHOR - 300,
        t_end=T_ANCHOR + 300,
        is_propeller=True,
    )
    path = tmp_path / "adsb.csv"
    write_adsb_csv(path, [track])

    source = LocalFileAdsbSource(path)
    tracks = source.load_tracks(_anchor_date_utc(), BBOX)

    assert len(tracks) == 1
    loaded = tracks[0]
    assert loaded.icao24 == "AAA111"
    assert loaded.is_propeller is True
    assert len(loaded.samples) == len(track.times)


def test_keeps_non_propeller_tracks_for_the_silence_filter(tmp_path):
    prop_track = make_straight_track(
        "AAA111", 47.0, -122.0, 1500.0, heading_deg=90.0, speed_mps=90.0,
        t_anchor=T_ANCHOR, t_start=T_ANCHOR - 60, t_end=T_ANCHOR + 60, is_propeller=True,
    )
    jet_track = make_straight_track(
        "JET999", 47.1, -122.1, 9000.0, heading_deg=200.0, speed_mps=230.0,
        t_anchor=T_ANCHOR, t_start=T_ANCHOR - 60, t_end=T_ANCHOR + 60, is_propeller=False,
    )
    path = tmp_path / "adsb.csv"
    write_adsb_csv(path, [prop_track, jet_track])

    source = LocalFileAdsbSource(path)
    tracks = source.load_tracks(_anchor_date_utc(), BBOX)

    # load_tracks must return everything in range -- filtering to
    # propeller-only happens later via adsb/filters.py, not here, so the
    # silence filter can still see the jet.
    assert len(tracks) == 2
    assert {t.is_propeller for t in tracks} == {True, False}


def test_sorts_and_deduplicates_out_of_order_rows(tmp_path):
    track = make_straight_track(
        "BBB222", 47.0, -122.0, 1500.0, heading_deg=0.0, speed_mps=100.0,
        t_anchor=T_ANCHOR, t_start=T_ANCHOR - 10, t_end=T_ANCHOR + 10, is_propeller=True, dt_s=1.0,
    )
    df = tracks_to_dataframe([track])
    duplicated_row = df.iloc[[5]]
    df_with_dup = pd.concat([df, duplicated_row], ignore_index=True)
    shuffled = df_with_dup.sample(frac=1.0, random_state=0).reset_index(drop=True)

    path = tmp_path / "shuffled.csv"
    shuffled.to_csv(path, index=False)

    source = LocalFileAdsbSource(path)
    tracks = source.load_tracks(_anchor_date_utc(), BBOX)

    assert len(tracks) == 1
    loaded_times = np.array([s.t for s in tracks[0].samples])
    assert np.all(np.diff(loaded_times) > 0)
    assert len(loaded_times) == len(track.times)  # duplicate collapsed, not kept as an extra sample


def test_splits_track_on_time_gap(tmp_path):
    track = make_straight_track(
        "CCC333", 47.0, -122.0, 1500.0, heading_deg=45.0, speed_mps=80.0,
        t_anchor=T_ANCHOR, t_start=T_ANCHOR - 100, t_end=T_ANCHOR + 100, is_propeller=True, dt_s=1.0,
    )
    df = tracks_to_dataframe([track])
    mid = len(df) // 2
    gapped = pd.concat([df.iloc[: mid - 40], df.iloc[mid + 40 :]], ignore_index=True)  # ~80s gap in the middle

    path = tmp_path / "gapped.csv"
    gapped.to_csv(path, index=False)

    source = LocalFileAdsbSource(path, max_track_gap_s=60.0)
    tracks = source.load_tracks(_anchor_date_utc(), BBOX)

    assert len(tracks) == 2
    assert all(t.icao24.startswith("CCC333") for t in tracks)
    assert {t.icao24 for t in tracks} == {"CCC333#0", "CCC333#1"}


def test_retains_flight_crossing_the_requested_date_via_tz_padding(tmp_path):
    next_day_midnight_utc = dt.datetime.combine(
        _anchor_date_utc() + dt.timedelta(days=1), dt.time.min, tzinfo=dt.timezone.utc
    ).timestamp()
    track = make_straight_track(
        "DDD444", 47.0, -122.0, 1500.0, heading_deg=0.0, speed_mps=90.0,
        t_anchor=next_day_midnight_utc,
        t_start=next_day_midnight_utc - 1800,
        t_end=next_day_midnight_utc + 1800,
        is_propeller=True,
        dt_s=5.0,
    )
    path = tmp_path / "midnight.csv"
    write_adsb_csv(path, [track])

    source = LocalFileAdsbSource(path)
    tracks = source.load_tracks(_anchor_date_utc(), BBOX, tz_pad_hours=12.0)

    assert len(tracks) == 1
