"""Wires the four pipeline stages behind a single geolocate() entry point.

**v1 limitation**: if more than two Doppler events are detected, only
the two chronologically-earliest are tried. They are, by construction,
chronologically adjacent (no third detected event lies between them),
which matters for the silence filter: picking a non-adjacent pair would
let a real detected event slip into the "nothing else was heard between
these two" window and the coarse filter would wrongly reject the true
location. Trying multiple (not necessarily adjacent) event pairs is
deferred.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from .adsb.filters import filter_aircraft_type
from .adsb.interface import AdsbSource, BBox
from .audio.events import find_doppler_curves
from .audio.fit import fit_doppler_event
from .audio.spectrogram import generate_spectrogram
from .grid.coarse import coarse_filter, generate_grid
from .grid.fine import locate, upscale_grid
from .physics.geodesy import DemSource, FlatDem
from .types import ScoredCandidate


def geolocate(
    audio_path: str | Path,
    date: dt.date,
    bbox: BBox,
    adsb_source: AdsbSource,
    dem: DemSource | None = None,
    coarse_resolution_m: float = 200.0,
    fine_resolution_m: float = 100.0,
    tz_pad_hours: float = 12.0,
    error_margin_s: float | None = None,
    d_min_tolerance_m: float | None = None,
    audibility_radius_m: float = 5000.0,
) -> ScoredCandidate:
    """Run the full pipeline: audio analysis -> ADS-B ingestion -> coarse
    grid filtering -> precision Doppler scoring. Raises ValueError if
    fewer than 2 Doppler events are detected, or if no candidate survives
    the coarse or fine stages.
    """
    dem = dem if dem is not None else FlatDem(0.0)

    spectrogram = generate_spectrogram(audio_path)
    ridges = find_doppler_curves(spectrogram)
    if len(ridges) < 2:
        raise ValueError(f"need at least 2 Doppler events to geolocate, found {len(ridges)}")
    detected_events = sorted(
        (fit_doppler_event(ridge, event_id=i) for i, ridge in enumerate(ridges)),
        key=lambda event: event.t_heard_ref,
    )
    event_1, event_2 = detected_events[0], detected_events[1]  # chronologically adjacent, see module docstring

    all_tracks = adsb_source.load_tracks(date, bbox, tz_pad_hours=tz_pad_hours)
    match_tracks = filter_aircraft_type(all_tracks, "propeller")

    coarse_points = generate_grid(bbox, coarse_resolution_m, dem)
    coarse_survivors = coarse_filter(
        coarse_points,
        (event_1, event_2),
        match_tracks=match_tracks,
        silence_tracks=all_tracks,
        resolution_m=coarse_resolution_m,
        error_margin_s=error_margin_s,
        d_min_tolerance_m=d_min_tolerance_m,
        audibility_radius_m=audibility_radius_m,
    )
    if not coarse_survivors:
        raise ValueError("no coarse candidates survived -- no location/flight-pair combination matched the audio")

    fine_candidates = upscale_grid(coarse_survivors, coarse_resolution_m, fine_resolution_m, dem)
    return locate(fine_candidates, (event_1, event_2))
