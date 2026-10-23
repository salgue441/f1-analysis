# ADR-0006: OpenF1 replaces FastF1 as the lap, tyre, weather and telemetry source

- **Status:** Accepted
- **Date:** 2026-10-22
- **Supersedes:** the FastF1 parts of ADR-0001/0003 and SAD §3.2

## Context
FastF1 reads the official F1 live-timing archive (`livetiming.formula1.com/static`). Since about 15 October 2026 every file there returns **HTTP 401** unless the request carries an F1 TV login, and FastF1's fallback mirror doesn't help without one. On a clean cache `Session.load()` returns no laps, weather or telemetry for **any** season (verified for 2019, 2023 and 2024). Upstream tracks this as [theOehrly/Fast-F1#978](https://github.com/theOehrly/Fast-F1/issues/978).

OpenF1 (`api.openf1.org/v1`) serves the same timing feed free and without an account, but only **from 2023 on**. Its free tier allows roughly 30 requests per minute.

## Decision
- **OpenF1 is the lap-level source.** Endpoints: `laps`, `stints`, `pit`, `position`, `race_control`, `weather`, `drivers`, plus `car_data` and `location` for telemetry. Bronze keeps one raw JSON response per (session, endpoint) as `bronze.openf1_<endpoint>`.
- **Silver schemas don't change.** `silver.openf1` produces exactly the columns the FastF1 transforms produce (a test enforces this), so gold, features and models don't know the source. The FastF1 loader and transforms stay registered as `fastf1_*` specs for when the archive is reachable again.
- **Lap-level scope becomes 2023–2025** (`lap_seasons` in config). Jolpica results stay 2018–2025 for dimensions, history and priors.
- OpenF1 has no round numbers, so sessions are matched to Jolpica rounds by nearest scheduled start (±36 h).
- Fields OpenF1 lacks are derived and documented in `silver/openf1.py`:
  - per-lap track status comes from race-control periods overlapping the lap;
  - `is_accurate` means a full sector split and not a pit in/out lap;
  - deleted laps are parsed from race-control messages;
  - session time counts from the scheduled session start.

## Alternatives considered
- **F1 TV token for FastF1:** keeps 2018+, but needs a paid subscription, a FastF1 patch and a personal credential in the pipeline. It isn't reproducible for a reviewer cloning the repo.
- **Jolpica lap times for 2018–2022:** lap times and positions only, with no compounds or stints. That isn't enough for RQ1 and would mix two lap sources of different fidelity.
- **Wait for an upstream fix:** it would block Phases 2–5 for an unknown time.

## Consequences
- RQ1–RQ3 train on 2023–2024 and test on 2025, all inside the 2022+ regulation era. The planned "within-era" check becomes moot, and cross-era claims are out of scope. This is logged in the research plan's deviations table.
- Fewer circuits have more than one season of data, so partial pooling in RQ1 matters more.
- The ingest rate limit is about 7 requests per race session, or 2×20 per session with telemetry. A full 2023–2025 backfill takes about 25 minutes without telemetry.
