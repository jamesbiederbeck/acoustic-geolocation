# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository currently contains only a design README (`README.md`) describing the algorithm — there is no implementation, build system, test suite, or dependency manifest yet. There are no commands to build, lint, or test because no code exists. When implementation begins, update this file with the actual commands (e.g. how to run the pipeline end-to-end, how to run a single test) and the language/tooling choice.

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
- **Emission-time vs. heard-time distinction**: all distance/velocity/Doppler calculations for a modeled curve must be evaluated at the aircraft's position at *emission* time, found by solving the implicit delay equation `t_heard = t_emitted + d(t_emitted)/c`, not simply at the position at the observation timestamp.
- **Radial velocity** is the aircraft velocity vector's component along the aircraft→observer unit vector at emission time.
- Geolocation requires real-world geospatial math (WGS84 ellipsoid, 3D slant range including terrain elevation via a DEM) — flat-earth or 2D-only distance approximations will produce wrong Doppler curves at typical aircraft altitudes/ranges.
- The two-stage coarse→fine grid search exists for performance: the coarse stage uses cheap closed-form CPA/timing filters to eliminate most of the search space before the expensive full Doppler-curve synthesis and spectrogram-overlap scoring runs on the fine grid.
