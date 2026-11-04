"""Shared dataclasses passed between pipeline stages.

All dataclasses here use `eq=False`: several carry ndarray fields, and the
auto-generated `__eq__`/`__hash__` from `frozen=True` either raises
("ambiguous truth value") or crashes (ndarrays are unhashable) the moment
anything tries to compare, dedupe, or hash one of these.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .physics.geodesy import geodetic_to_ecef


@dataclass(eq=False)
class DopplerEvent:
    """A single Doppler curve segmented out of a recording's spectrogram."""

    event_id: int
    t_heard_start: float
    t_heard_end: float
    t_heard_ref: float  # heard time of the f = f0 crossing, where v_r = 0 -- the canonical event time
    t_heard_ref_err: float
    f0_estimate: float
    f0_estimate_err: float
    v_estimate: float
    v_estimate_err: float
    d_min_estimate: float
    d_min_estimate_err: float
    curve_times: np.ndarray
    curve_freqs: np.ndarray


@dataclass(eq=False)
class TrackSample:
    """One ADS-B state-vector report."""

    t: float  # absolute UTC epoch seconds
    lat: float
    lon: float
    alt_m: float  # treated as ellipsoidal height; see geodesy.py caveat
    ground_speed_mps: float | None = None
    track_deg: float | None = None
    vertical_rate_mps: float | None = None


@dataclass(eq=False)
class FlightTrack:
    """A single aircraft's time-ordered samples, with ECEF position/velocity
    interpolation suitable for `physics.propagation.solve_emission_time`.

    Position is linearly interpolated in ECEF (not lat/lon/alt) between
    samples, and velocity is the exact piecewise-constant derivative of
    that same interpolant -- `solve_emission_time`'s Newton step depends
    on velocity truly being d/dt of position, and `grid/coarse.py`'s
    closed-form per-segment CPA assumes linear segments.

    Samples must be time-sorted with strictly increasing, unique
    timestamps and at least 2 samples (enforced by the loader, not here).
    Queries beyond `extrapolation_margin_s` past either end return NaN;
    within that margin, position/velocity extrapolate linearly using the
    nearest end segment's velocity. The margin exists because sound
    emitted near a track's last sample is often *heard* well after it
    (propagation delay), and `solve_emission_time`'s seed evaluates
    position at the heard time itself.
    """

    icao24: str
    callsign: str | None
    aircraft_type: str | None
    is_propeller: bool
    samples: tuple[TrackSample, ...]
    extrapolation_margin_s: float = 60.0

    _times: np.ndarray = field(init=False, repr=False)
    _ecef: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._times = np.array([s.t for s in self.samples], dtype=float)
        self._ecef = geodetic_to_ecef(
            np.array([s.lat for s in self.samples]),
            np.array([s.lon for s in self.samples]),
            np.array([s.alt_m for s in self.samples]),
        )

    @property
    def t_start(self) -> float:
        return float(self._times[0])

    @property
    def t_end(self) -> float:
        return float(self._times[-1])

    @property
    def sample_times(self) -> np.ndarray:
        """Raw sample timestamps (seconds), time-sorted, for closed-form
        per-segment CPA computation in grid/coarse.py."""
        return self._times.copy()

    @property
    def sample_positions_ecef(self) -> np.ndarray:
        """Raw sample positions in ECEF, aligned with `sample_times`."""
        return self._ecef.copy()

    def _segment_index(self, t_flat: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self._times, t_flat, side="right")
        return np.clip(idx, 1, len(self._times) - 1)

    def _out_of_margin(self, t_flat: np.ndarray) -> np.ndarray:
        return (t_flat < self._times[0] - self.extrapolation_margin_s) | (
            t_flat > self._times[-1] + self.extrapolation_margin_s
        )

    def position_ecef(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        flat = t_arr.ravel()
        idx = self._segment_index(flat)
        t0, t1 = self._times[idx - 1], self._times[idx]
        v0, v1 = self._ecef[idx - 1], self._ecef[idx]
        slope = (v1 - v0) / (t1 - t0)[:, None]
        positions = v0 + slope * (flat - t0)[:, None]
        positions[self._out_of_margin(flat)] = np.nan
        return positions.reshape(t_arr.shape + (3,))

    def velocity_ecef(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        flat = t_arr.ravel()
        idx = self._segment_index(flat)
        t0, t1 = self._times[idx - 1], self._times[idx]
        v0, v1 = self._ecef[idx - 1], self._ecef[idx]
        slope = (v1 - v0) / (t1 - t0)[:, None]
        velocities = np.broadcast_to(slope, (len(flat), 3)).copy()
        velocities[self._out_of_margin(flat)] = np.nan
        return velocities.reshape(t_arr.shape + (3,))


@dataclass(eq=False)
class GridPoint:
    """A candidate observer location."""

    lat: float
    lon: float
    elevation_m: float
    ecef: np.ndarray  # cached (3,)


@dataclass(eq=False)
class Candidate:
    """A grid point paired with the two flights hypothesized to have
    produced the two detected Doppler events there, surviving the coarse
    time-offset, distance, and silence filters."""

    point: GridPoint
    flight_1: FlightTrack
    flight_2: FlightTrack
    t0_hypothesis: float  # absolute UTC epoch seconds for the recording's start
    coarse_time_offset_error: float
