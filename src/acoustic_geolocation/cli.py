"""`acoustic-geolocate` CLI entrypoint.

`--error-margin-s`/`--d-min-tolerance-m` default to None (auto-sized by
grid/coarse.py from the grid resolution and the audio stage's own
measurement error) -- a fixed default tight enough to feel sensible at
one resolution can reject the true grid cell outright at another, so
there is no single safe fixed default to hardcode here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from .adsb.interface import BBox
from .adsb.local_file import LocalFileAdsbSource
from .audio.events import find_doppler_curves
from .audio.fit import fit_doppler_event
from .audio.spectrogram import generate_spectrogram
from .physics.geodesy import DemSource, FlatDem
from .pipeline import geolocate


def _parse_bbox(value: str) -> BBox:
    parts = [float(x) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be lat_min,lon_min,lat_max,lon_max")
    return BBox(*parts)


def _parse_dem(value: str) -> DemSource:
    if value.startswith("flat:"):
        return FlatDem(float(value[len("flat:") :]))
    raise argparse.ArgumentTypeError("only 'flat:<elevation_m>' DEMs are supported in v1")


def _cmd_locate(args: argparse.Namespace) -> int:
    result = geolocate(
        args.audio_file,
        args.date,
        args.bbox,
        LocalFileAdsbSource(args.adsb),
        dem=_parse_dem(args.dem),
        coarse_resolution_m=args.coarse_resolution_m,
        fine_resolution_m=args.fine_resolution_m,
        tz_pad_hours=args.tz_pad_hours,
        error_margin_s=args.error_margin_s,
        d_min_tolerance_m=args.d_min_tolerance_m,
        audibility_radius_m=args.audibility_radius_m,
    )
    payload = {
        "lat": result.candidate.point.lat,
        "lon": result.candidate.point.lon,
        "elevation_m": result.candidate.point.elevation_m,
        "score": result.score,
        "t0_refined": result.t0_refined,
    }
    if args.output == "json":
        print(json.dumps(payload))
    else:
        print(
            f"lat={payload['lat']:.6f} lon={payload['lon']:.6f} elevation_m={payload['elevation_m']:.1f} "
            f"score={payload['score']:.4f} t0={payload['t0_refined']:.3f}"
        )
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    spectrogram = generate_spectrogram(args.audio_file)
    ridges = find_doppler_curves(spectrogram)
    events = [fit_doppler_event(ridge, event_id=i) for i, ridge in enumerate(ridges)]
    payload = [
        {
            "event_id": event.event_id,
            "t_heard_ref": event.t_heard_ref,
            "f0_estimate": event.f0_estimate,
            "v_estimate": event.v_estimate,
            "d_min_estimate": event.d_min_estimate,
        }
        for event in events
    ]
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acoustic-geolocate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    locate_parser = subparsers.add_parser("locate", help="Geolocate a recording from background aircraft audio")
    locate_parser.add_argument("audio_file")
    locate_parser.add_argument("date", type=dt.date.fromisoformat, help="UTC date to search, YYYY-MM-DD")
    locate_parser.add_argument("bbox", type=_parse_bbox, help="lat_min,lon_min,lat_max,lon_max")
    locate_parser.add_argument("--adsb", required=True, help="Local ADS-B dump file (CSV or JSON)")
    locate_parser.add_argument("--tz-pad-hours", type=float, default=12.0)
    locate_parser.add_argument("--coarse-resolution-m", type=float, default=200.0)
    locate_parser.add_argument("--fine-resolution-m", type=float, default=100.0)
    locate_parser.add_argument("--error-margin-s", type=float, default=None)
    locate_parser.add_argument("--d-min-tolerance-m", type=float, default=None)
    locate_parser.add_argument("--audibility-radius-m", type=float, default=5000.0)
    locate_parser.add_argument("--dem", default="flat:0.0", help="'flat:<elevation_m>' (only form supported in v1)")
    locate_parser.add_argument("--output", choices=["json", "text"], default="text")
    locate_parser.set_defaults(func=_cmd_locate)

    events_parser = subparsers.add_parser("events", help="Run the audio stage only and print detected events")
    events_parser.add_argument("audio_file")
    events_parser.set_defaults(func=_cmd_events)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
