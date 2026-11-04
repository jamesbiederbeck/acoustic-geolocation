# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Naming

The project is named `acoustic_geolocation`, not "MLAT". MLAT (multilateration) normally means locating a *transmitter* via time-difference-of-arrival across multiple spatially-separated receivers. This project does the opposite: a single recording's Doppler curves are matched against known ADS-B flight tracks to locate the *receiver*. If the project is ever renamed again, keep it reflecting single-recording Doppler-curve matching, not multilateration.

## Project status

`physics/` (geodesy, Doppler, and the propagation-delay solver) is implemented and tested. The four pipeline stages themselves (`audio/`, `adsb/`, `grid/coarse.py`, `grid/fine.py`) and `pipeline.py`/`cli.py` are still unimplemented stubs. Commands:

- Install (editable, with dev deps): `pip install -e '.[dev]'`
- Run tests: `pytest`
- Run a single test: `pytest tests/test_physics_propagation.py::test_name`

As stages land, this section should track what's actually implemented vs. stubbed.

## Versioning

- Maintain a `VERSION` file at the repo root containing a single semver string (e.g. `0.0.1`), starting at `0.0.1`.
- Bump `VERSION` on any commit that changes code or algorithm behavior (not for pure typo/doc fixes unrelated to versioned behavior): patch for fixes, minor for backward-compatible additions, major for breaking changes.
- Every commit that bumps `VERSION` must, in the same commit, update `CLAUDE.md` to reflect the current state of the codebase (new commands, architecture changes, new invariants, etc.) — the two files should never drift apart.

## Purpose

The goal is to geolocate a video/audio recording by extracting the Doppler-shifted sound of background aircraft from its spectrogram and matching that acoustic signature against historical ADS-B flight-track data, narrowing down to the exact location the recording was made.

## Architecture (per README.md design)

The pipeline has four conceptual stages, intended to be implemented in this order since each depends on the output of the previous one:

1. **Audio Analysis** — generate a spectrogram from the input recording and extract discrete Doppler "events" (curves), plus the time offset between events and rough speed/distance ranges per event.
2. **ADS-B ingestion** — fetch historical flight data for the date and rough search area, filtered to aircraft types whose acoustic signature is modeled (e.g. propeller aircraft).
3. **Coarse grid filtering** — over a coarse geospatial grid (~200m), compute the delayed closest-point-of-approach (CPA) per candidate point per event, accounting for sound travel time (not just flight time), and keep only points whose predicted inter-event time offset matches the measured one *and* whose predicted "silence" (no other audible flights between events) matches the recording.
4. **Precision Doppler modeling & scoring** — upscale surviving coarse points to a fine grid (~100m), generate a theoretical Doppler curve per candidate point by integrating emission-time vs. observation-time (sound emitted at `t_emitted` is heard at `t_emitted + distance(t_emitted)/c`, so the model must solve for the aircraft's position *at emission time*, not at heard time), and score the modeled curve against spectrogram brightness. The highest-scoring point is the answer.

Key invariants to preserve when implementing this:
- **Emission-time vs. heard-time distinction**: all distance/velocity/Doppler calculations for a modeled curve must be evaluated at the aircraft's position at *emission* time, found by solving the implicit delay equation `t_heard = t_emitted + d(t_emitted)/c`, not simply at the position at the observation timestamp. The README's pseudocode line `emission_time = t - distance(position_at(t))/c` is only the *first* fixed-point iteration of that solve, not a converged answer — never copy it verbatim as the final computation.
- **Radial velocity** is the aircraft velocity vector's component along the aircraft→observer unit vector at emission time.
- **The recording's absolute start time (`T0`) is unknown** and must be solved for jointly with location — treat it as a nuisance parameter fit alongside each candidate point, not a given input.
- **Each detected Doppler event's canonical time is the `f = f0` crossing** (the curve's inflection point, where radial velocity is zero) — not the event's start, peak, or midpoint — because that is exactly the instant the delayed-CPA computation predicts.
- **The coarse grid stage needs no root-finding.** For subsonic sources, heard-time is monotonic in emission-time, so the coarse delayed CPA is a closed-form/argmin minimization of slant distance over emission time, then `t_heard = t_e* + d(t_e*)/c`. The implicit delay equation only needs iterative solving (fixed-point or similar) in fine-curve synthesis and the silence filter, which must evaluate emission time at arbitrary heard-times rather than just at the CPA.
- Geolocation requires real-world geospatial math (WGS84 ellipsoid, 3D slant range including terrain elevation via a DEM) — flat-earth or 2D-only distance approximations will produce wrong Doppler curves at typical aircraft altitudes/ranges.
- The two-stage coarse→fine grid search exists for performance: the coarse stage uses cheap closed-form CPA/timing filters to eliminate most of the search space before the expensive full Doppler-curve synthesis and spectrogram-overlap scoring runs on the fine grid.
- v1 treats barometric, orthometric (DEM/MSL), and ellipsoidal heights as equivalent — a known, documented approximation, not an oversight; a geoid-undulation correction is deferred.
- `physics/propagation.py::solve_emission_time` takes plain `position_fn`/`velocity_fn` callables (time array -> ECEF array), not a `FlightTrack` object — `physics/` has no dependency on `adsb`/`types`. `FlightTrack` (added in the ADS-B ingestion stage) must expose `position_ecef(t)`/`velocity_ecef(t)` methods matching that signature, including returning NaN rows for times outside the track's sample range (callers of `solve_emission_time` rely on NaN propagating to a NaN, unconverged result rather than raising).
- Newton's method, not fixed-point iteration, is used for the propagation-delay solve — fixed-point contracts at rate `|v_r|/c` and needs too many iterations for fast aircraft; Newton uses the exact derivative `1 - v_r/c` and converges in ~4-5 iterations.
