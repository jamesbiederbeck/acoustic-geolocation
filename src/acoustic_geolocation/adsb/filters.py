"""Aircraft-type filtering, applied *after* ADS-B ingestion.

This narrows tracks down to the aircraft types whose acoustic signature
is modeled (e.g. propeller-only) for *matching* against audio events. It
must never be applied before the silence filter in `grid/coarse.py`: a
jet (or any other type) is still capable of being heard, and so must
remain in the "was anything else audible between the two events" check
even though it isn't itself a match candidate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from ..types import FlightTrack

Predicate = Literal["propeller"]


def filter_aircraft_type(
    tracks: list[FlightTrack], predicate: Predicate | Callable[[FlightTrack], bool]
) -> list[FlightTrack]:
    if predicate == "propeller":
        predicate_fn: Callable[[FlightTrack], bool] = lambda track: track.is_propeller
    elif callable(predicate):
        predicate_fn = predicate
    else:
        raise ValueError(f"Unknown aircraft filter predicate: {predicate!r}")
    return [track for track in tracks if predicate_fn(track)]
