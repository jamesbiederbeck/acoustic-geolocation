#!/usr/bin/env python3
"""Large-scale validation harness: simulate a couple of days of
propeller traffic over a Washington-state-sized airspace (climbing away
from real WA airports, cruising, descending into another), then at many
random (location, time) trials synthesize what a recording there would
have picked up -- via the same `audibility_radius_m` concept the
pipeline's own silence filter uses -- and run the real `geolocate()`
pipeline against it, reporting how often and how accurately it recovers
the truth.

This is a dev/validation tool, not part of the installed package (see
CLAUDE.md's note on why synthetic generators live under tests/fixtures/
and scripts/, not src/).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import signal
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acoustic_geolocation.adsb.interface import BBox
from acoustic_geolocation.adsb.local_file import LocalFileAdsbSource
from acoustic_geolocation.physics.geodesy import FlatDem, ecef_to_geodetic, geodetic_to_ecef
from acoustic_geolocation.pipeline import geolocate
from acoustic_geolocation.types import GridPoint

from tests.fixtures.airports import WASHINGTON_AIRPORTS  # noqa: E402
from tests.fixtures.simulate_airspace import epoch, generate_world  # noqa: E402
from tests.fixtures.simulate_recording import audible_tracks_in_window, synthesize_recording_at  # noqa: E402
from tests.fixtures.synth_tracks import to_flight_track, write_adsb_csv  # noqa: E402

_METERS_PER_DEGREE_LAT = 111_320.0
WA_BBOX = BBox(lat_min=45.5, lon_min=-124.8, lat_max=49.0, lon_max=-116.9)


class _TrialTimeout(Exception):
    pass


@contextlib.contextmanager
def _trial_timeout(seconds: float):
    """Guards a single trial's `geolocate()` call against running far
    longer than the rest. `grid/coarse.py::coarse_filter` calls
    `delayed_cpa` twice per (point, ordered-flight-pair) combination via
    `itertools.permutations` -- O(n_points * n_tracks^2) -- which the
    small hand-built fixtures the per-stage tests use never stress, but
    this harness's locally-dense, real-world-shaped traffic can (measured
    at ~1.28s per 100 coarse points with a dozen local tracks, i.e.
    tens of seconds once the grid itself is a few thousand points). This
    is a real coarse_filter cost-scaling characteristic, not confirmed to
    be a bug, and not something this harness changes (that lives in
    `src/`); the per-trial alarm here just keeps one such trial from
    blocking the whole run, counting it as a `timeout` outcome instead.
    """

    def _raise(signum, frame):
        raise _TrialTimeout()

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _local_search_bbox(rng, lat: float, lon: float, pad_m: float) -> BBox:
    """A `pad_m`-radius bbox *roughly* around (lat, lon) -- the "rough
    search area" a real user would supply (per the README/CLAUDE.md
    architecture description), not the full WA_BBOX, nor a box perfectly
    centered on the truth. Searching the *entire* Washington-sized bbox
    at `--coarse-resolution-m` would be millions of grid points; cropping
    to a local area around the trial's truth location keeps each trial's
    coarse grid search tractable while the underlying simulated *dataset*
    still spans all of WA.

    The center is randomly offset (observer stays within the box with
    >= 40% margin) rather than placed exactly at (lat, lon): a bbox
    perfectly centered on the truth, with a pad that's an exact multiple
    of the coarse resolution, would otherwise put the true location
    exactly on a grid node every time, never exercising the grid
    quantization error the coarse stage's tolerances are built to absorb.
    """
    lat_pad = pad_m / _METERS_PER_DEGREE_LAT
    meters_per_degree_lon = _METERS_PER_DEGREE_LAT * max(np.cos(np.radians(lat)), 1e-6)
    lon_pad = pad_m / meters_per_degree_lon

    max_shift_frac = 0.6
    lat_center = lat + rng.uniform(-max_shift_frac, max_shift_frac) * lat_pad
    lon_center = lon + rng.uniform(-max_shift_frac, max_shift_frac) * lon_pad
    return BBox(
        lat_min=lat_center - lat_pad, lon_min=lon_center - lon_pad,
        lat_max=lat_center + lat_pad, lon_max=lon_center + lon_pad,
    )


def _build_encounter_index(tracks, duration_s, bucket_deg=0.015, sample_dt_s=15.0, min_gap_s=15.0):
    """Precompute candidate (t0, lat, lon) trials: a location two
    different tracks both pass near, at *offset* times (not the same
    moment) that both fit inside one `duration_s` recording window.

    This deliberately looks for staggered passes, not simultaneous ones:
    `audio/events.py::find_doppler_curves` keeps only the single loudest
    frequency bin per time frame (a documented v1 limitation), so two
    aircraft that are both near their loudest at the very same instant
    just stomp on each other in that detector, while two passes offset
    by tens of seconds -- each dominating the spectrum during its own
    sub-window -- are exactly the case the pipeline is built to separate
    via the `f = f0` crossing time of each. Blind random (location, time)
    guessing over a WA-sized, multi-day, sparse simulated airspace
    essentially never lands on a usable pair; this direct spatial-bucket
    search is what makes trial generation tractable at all.
    """
    buckets: dict[tuple[int, int], list[tuple[int, float, float, float]]] = {}
    for track_idx, track in enumerate(tracks):
        times = np.arange(track.t_start, track.t_end, sample_dt_s)
        if len(times) == 0:
            continue
        lat, lon, _alt = ecef_to_geodetic(track.position_ecef(times))
        for t, la, lo in zip(times, lat, lon):
            key = (int(round(la / bucket_deg)), int(round(lo / bucket_deg)))
            buckets.setdefault(key, []).append((track_idx, float(t), float(la), float(lo)))

    max_gap_s = duration_s - 30.0
    encounters = []
    for entries in buckets.values():
        by_track: dict[int, list[tuple[float, float, float]]] = {}
        for idx, t, la, lo in entries:
            by_track.setdefault(idx, []).append((t, la, lo))
        track_ids = list(by_track)
        for i in range(len(track_ids)):
            ti, lai, loi = by_track[track_ids[i]][len(by_track[track_ids[i]]) // 2]
            for j in range(i + 1, len(track_ids)):
                tj, laj, loj = by_track[track_ids[j]][len(by_track[track_ids[j]]) // 2]
                gap_s = abs(ti - tj)
                if min_gap_s <= gap_s <= max_gap_s:
                    t0 = min(ti, tj) - 15.0
                    encounters.append((t0, 0.5 * (lai + laj), 0.5 * (loi + loj)))
    return encounters


def _pick_trial(rng, encounters, tracks, duration_s, audibility_radius_m, max_attempts=50, max_audible=4):
    """Sample a trial (observer, t0) from the precomputed `encounters`
    index, jittered a little in both position and start time for
    variety, then verify it actually has between 2 and `max_audible`
    tracks audible at some point during the window -- a sanity re-check,
    since the jitter can push a borderline encounter back out of range.
    """
    if not encounters:
        return None

    for _ in range(max_attempts):
        t0_anchor, lat, lon = encounters[rng.integers(len(encounters))]
        t0 = t0_anchor + rng.uniform(-10.0, 10.0)

        offset_m = rng.uniform(0.0, 0.3 * audibility_radius_m)
        bearing = rng.uniform(0.0, 2 * np.pi)
        lat_obs = lat + (offset_m * np.cos(bearing)) / _METERS_PER_DEGREE_LAT
        meters_per_degree_lon = _METERS_PER_DEGREE_LAT * max(np.cos(np.radians(lat_obs)), 1e-6)
        lon_obs = lon + (offset_m * np.sin(bearing)) / meters_per_degree_lon
        elev_obs = float(rng.uniform(0.0, 400.0))

        observer = GridPoint(lat=lat_obs, lon=lon_obs, elevation_m=elev_obs, ecef=geodetic_to_ecef(lat_obs, lon_obs, elev_obs))
        audible = audible_tracks_in_window(observer, tracks, (t0, t0 + duration_s), audibility_radius_m)
        if 2 <= len(audible) <= max_audible:
            return observer, t0
    return None


def run(args: argparse.Namespace) -> dict[str, list[dict]]:
    rng = np.random.default_rng(args.seed)
    t_start = epoch(args.date)
    t_end = t_start + args.days * 86400.0

    print(f"Generating {args.n_flights} flights over {args.days} day(s)...")
    gen_start = time.time()
    synth_tracks, f0_by_icao24 = generate_world(rng, WASHINGTON_AIRPORTS, t_start, t_end, args.n_flights)
    tracks = [to_flight_track(t) for t in synth_tracks]
    print(f"  {len(tracks)} tracks, {sum(len(t.samples) for t in tracks)} total samples, {time.time() - gen_start:.1f}s")

    adsb_path = Path(args.adsb_out) if args.adsb_out else Path(tempfile.mkstemp(suffix=".csv")[1])
    write_adsb_csv(adsb_path, synth_tracks)
    print(f"  ADS-B dump written to {adsb_path}")

    adsb_source = LocalFileAdsbSource(adsb_path)

    recording_duration_s = 120.0

    print("Indexing close-encounter candidates...")
    index_start = time.time()
    encounters = _build_encounter_index(tracks, recording_duration_s)
    print(f"  {len(encounters)} candidate encounters, {time.time() - index_start:.1f}s")
    outcomes: dict[str, list[dict]] = {
        "ok": [], "wrong_location": [], "insufficient_events": [], "no_candidates": [],
        "no_valid_trial_location": [], "timeout": [], "exception": [],
    }

    print(f"Running {args.n_trials} trials...")
    trial_start = time.time()
    for trial_idx in range(args.n_trials):
        trial_t1 = time.time()
        picked = _pick_trial(rng, encounters, tracks, recording_duration_s, args.audibility_radius_m)
        if picked is None:
            outcomes["no_valid_trial_location"].append({})
            print(f"  [{trial_idx + 1}/{args.n_trials}] no_valid_trial_location")
            continue
        observer, t0 = picked

        try:
            waveform, samplerate = synthesize_recording_at(
                observer, t0, recording_duration_s, tracks, args.audibility_radius_m, f0_by_icao24=f0_by_icao24,
            )
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav:
                sf.write(tmp_wav.name, waveform, samplerate)
                recording_date = dt.datetime.fromtimestamp(t0, tz=dt.timezone.utc).date()
                search_bbox = _local_search_bbox(rng, observer.lat, observer.lon, args.search_radius_m)
                # DEM must reflect the true observer's elevation here -- the
                # pipeline's own grid stage looks up each candidate point's
                # elevation from this same DemSource, so a fixed FlatDem(0.0)
                # would silently put every candidate below the real
                # observer, biasing every trial's slant-range geometry.
                dem = FlatDem(observer.elevation_m)
                with _trial_timeout(args.trial_timeout_s):
                    result = geolocate(
                        tmp_wav.name, recording_date, search_bbox, adsb_source, dem=dem,
                        coarse_resolution_m=args.coarse_resolution_m, fine_resolution_m=args.fine_resolution_m,
                        audibility_radius_m=args.audibility_radius_m, tz_pad_hours=args.tz_pad_hours,
                    )
        except _TrialTimeout:
            outcomes["timeout"].append({"trial": trial_idx})
            print(f"  [{trial_idx + 1}/{args.n_trials}] timeout (>{args.trial_timeout_s:.0f}s)")
            continue
        except ValueError as exc:
            message = str(exc)
            if "need at least 2 Doppler events" in message:
                category = "insufficient_events"
            elif "no coarse" in message or "no fine" in message or "survived" in message:
                category = "no_candidates"
            else:
                category = "exception"
            outcomes[category].append({"trial": trial_idx, "error": message, "trace": traceback.format_exc()})
            print(f"  [{trial_idx + 1}/{args.n_trials}] {category} ({time.time() - trial_t1:.1f}s)")
            continue
        except Exception as exc:  # noqa: BLE001 -- the harness must keep going and tally this, not crash
            outcomes["exception"].append({"trial": trial_idx, "error": str(exc), "trace": traceback.format_exc()})
            print(f"  [{trial_idx + 1}/{args.n_trials}] exception ({time.time() - trial_t1:.1f}s): {exc}")
            continue

        dlat_m = (result.candidate.point.lat - observer.lat) * _METERS_PER_DEGREE_LAT
        dlon_m = (result.candidate.point.lon - observer.lon) * _METERS_PER_DEGREE_LAT * np.cos(np.radians(observer.lat))
        distance_m = float(np.hypot(dlat_m, dlon_m))
        t0_error_s = abs(result.t0_refined - t0)

        record = {"trial": trial_idx, "distance_m": distance_m, "t0_error_s": t0_error_s, "score": result.score}
        category = "ok" if distance_m <= 2 * args.fine_resolution_m and t0_error_s <= 1.0 else "wrong_location"
        outcomes[category].append(record)
        print(
            f"  [{trial_idx + 1}/{args.n_trials}] {category} "
            f"(distance={distance_m:.1f}m, t0_err={t0_error_s:.2f}s, {time.time() - trial_t1:.1f}s)"
        )

    elapsed = time.time() - trial_start
    print(f"\n=== Validation harness report ({elapsed:.1f}s for {args.n_trials} trials) ===")
    for category, records in outcomes.items():
        print(f"  {category}: {len(records)}")
    if outcomes["ok"]:
        distances = [r["distance_m"] for r in outcomes["ok"]]
        t0_errors = [r["t0_error_s"] for r in outcomes["ok"]]
        print(f"  ok location error: median={np.median(distances):.1f}m p95={np.percentile(distances, 95):.1f}m")
        print(f"  ok T0 error: median={np.median(t0_errors):.3f}s p95={np.percentile(t0_errors, 95):.3f}s")

    return outcomes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-flights", type=int, default=1000)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--days", type=float, default=2.0)
    parser.add_argument("--date", type=dt.date.fromisoformat, default=dt.date(2026, 6, 15))
    parser.add_argument("--audibility-radius-m", type=float, default=5000.0)
    parser.add_argument(
        "--search-radius-m", type=float, default=6_000.0,
        help="Per-trial local search-bbox half-width -- kept close to --audibility-radius-m since the coarse grid "
        "search's cost scales with (search area / coarse_resolution_m^2) x (local track count)^2",
    )
    parser.add_argument("--coarse-resolution-m", type=float, default=200.0)
    parser.add_argument("--fine-resolution-m", type=float, default=100.0)
    parser.add_argument("--tz-pad-hours", type=float, default=1.0)
    parser.add_argument("--trial-timeout-s", type=float, default=45.0, help="Per-trial geolocate() watchdog")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adsb-out", default=None, help="Path to persist the generated ADS-B CSV (default: a temp file)")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
