"""Local CSV/JSON ADS-B dump loader implementing AdsbSource.

v1 expects an explicit `is_propeller` column in the dump rather than
deriving it from an ICAO aircraft-type-designator lookup table -- that
lookup is deferred; for now the data source (or, in tests, the synthetic
fixture generator) is responsible for labeling it.

Rows are sorted by timestamp and deduplicated per `icao24` before use:
`FlightTrack`'s linear-ECEF interpolation assumes strictly increasing,
unique timestamps, and a real-world dump assembled from multiple
receivers can arrive unsorted with duplicate reports. A single `icao24`
is also split into separate `FlightTrack`s wherever consecutive samples
are more than `max_track_gap_s` apart, so interpolation never bridges a
coverage dropout (or two unrelated flights that happen to reuse an
address).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ..types import FlightTrack, TrackSample
from .interface import BBox

_REQUIRED_COLUMNS = {"icao24", "timestamp", "lat", "lon", "alt_m", "is_propeller"}
_OPTIONAL_SAMPLE_COLUMNS = {"gs": "ground_speed_mps", "track": "track_deg", "vrate": "vertical_rate_mps"}


class LocalFileAdsbSource:
    """AdsbSource backed by a local CSV (or JSON-records) dump file."""

    def __init__(self, path: str | Path, max_track_gap_s: float = 60.0) -> None:
        self._path = Path(path)
        self._max_track_gap_s = max_track_gap_s

    def load_tracks(self, date: dt.date, bbox: BBox, *, tz_pad_hours: float = 12.0) -> list[FlightTrack]:
        df = pd.read_json(self._path) if self._path.suffix == ".json" else pd.read_csv(self._path)

        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"ADS-B dump {self._path} missing required columns: {sorted(missing)}")

        window_start = dt.datetime.combine(date, dt.time.min, tzinfo=dt.timezone.utc) - dt.timedelta(
            hours=tz_pad_hours
        )
        window_end = dt.datetime.combine(date, dt.time.max, tzinfo=dt.timezone.utc) + dt.timedelta(
            hours=tz_pad_hours
        )
        df = df[(df["timestamp"] >= window_start.timestamp()) & (df["timestamp"] <= window_end.timestamp())]
        df = df[
            (df["lat"] >= bbox.lat_min)
            & (df["lat"] <= bbox.lat_max)
            & (df["lon"] >= bbox.lon_min)
            & (df["lon"] <= bbox.lon_max)
        ]

        tracks: list[FlightTrack] = []
        for icao24, group in df.groupby("icao24"):
            group = group.drop_duplicates(subset="timestamp").sort_values("timestamp")
            times = group["timestamp"].to_numpy(dtype=float)
            if len(times) < 2:
                continue

            gap_breaks = np.where(np.diff(times) > self._max_track_gap_s)[0]
            segment_starts = np.concatenate([[0], gap_breaks + 1])
            segment_ends = np.concatenate([gap_breaks + 1, [len(times)]])
            multi_segment = len(segment_starts) > 1

            for seg_idx, (start_idx, end_idx) in enumerate(zip(segment_starts, segment_ends)):
                segment = group.iloc[start_idx:end_idx]
                if len(segment) < 2:
                    continue
                tracks.append(self._build_track(icao24, seg_idx if multi_segment else None, segment))

        return tracks

    @staticmethod
    def _build_track(icao24: object, seg_idx: int | None, segment: pd.DataFrame) -> FlightTrack:
        samples = tuple(
            TrackSample(
                t=float(row.timestamp),
                lat=float(row.lat),
                lon=float(row.lon),
                alt_m=float(row.alt_m),
                **{
                    attr: float(getattr(row, col))
                    for col, attr in _OPTIONAL_SAMPLE_COLUMNS.items()
                    if col in segment.columns and not pd.isna(getattr(row, col))
                },
            )
            for row in segment.itertuples()
        )
        return FlightTrack(
            icao24=f"{icao24}#{seg_idx}" if seg_idx is not None else str(icao24),
            callsign=str(segment.iloc[0]["callsign"]) if "callsign" in segment.columns else None,
            aircraft_type=str(segment.iloc[0]["aircraft_type"]) if "aircraft_type" in segment.columns else None,
            is_propeller=bool(segment.iloc[0]["is_propeller"]),
            samples=samples,
        )
