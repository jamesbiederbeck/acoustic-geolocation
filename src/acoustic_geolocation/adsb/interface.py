"""AdsbSource protocol: returns FlightTracks for a date + bounding box.

Implementations are swappable: `LocalFileAdsbSource` (this commit) reads a
local CSV/JSON dump; a future live-API client (e.g. OpenSky) would
implement the same protocol with no change needed in `pipeline.py`/
`cli.py`, which inject an `AdsbSource` instance rather than importing a
concrete implementation directly.

`load_tracks` returns *all* tracks in range, not just aircraft-type
matches -- the silence filter in `grid/coarse.py` needs to know about
every audible aircraft (a jet overhead between two events is still
audible), not just the propeller aircraft being matched against audio
events. Narrow to matchable types afterward with `filters.py`.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    from ..types import FlightTrack


class BBox(NamedTuple):
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float


class AdsbSource(Protocol):
    def load_tracks(self, date: dt.date, bbox: BBox, *, tz_pad_hours: float = 12.0) -> list[FlightTrack]: ...
