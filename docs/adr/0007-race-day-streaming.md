# ADR-0007: Race-day streaming with Structured Streaming over landing files

- **Status:** Accepted
- **Date:** 2026-10-23

## Context
Phase 7 adds a live view of a race: positions, gaps, tyres and strategy signals, updated within seconds. OpenF1 publishes live data over MQTT and WebSockets, but only on its paid Sponsor tier; historical data stays free. The pipeline must be buildable, testable and demonstrable without a subscription, and must work unchanged once one is added.

## Decision
- **Producers write landing files; one Spark query consumes them.** A producer writes JSON-lines files, one OpenF1 record per line plus `_endpoint` and `_emitted_at`, into `data/landing/openf1/<session_key>/`. It writes under a hidden name and renames when complete, so Spark never reads a partial file. There are two producers:
  - **replay** (default): reads a past race from `bronze.openf1_*` and releases each record when it would have been published (a lap when it ends, a stint when its first lap starts), on a clock running `speed`× real time;
  - **live**: subscribes to OpenF1's MQTT topics with the OAuth2 token as password and re-authenticates before the hourly expiry. It needs `OPENF1_USERNAME` / `OPENF1_PASSWORD` and the optional `live` extra.
- **Spark Structured Streaming, file source, `foreachBatch`.** Each micro-batch appends the raw records to `stream.events`, rebuilds `stream.race_state` (one row per car) for the session, and logs timings to `stream._batches`.
- **The race state is rebuilt, not updated incrementally.** A race is a few thousand records, so rebuilding is cheap. It also reuses the batch layers' pure functions: the OpenF1 record schemas and track-status coding from `silver.openf1`. The session's log is kept in memory across batches, so a batch costs the same late in the race as early. Delta is read once, to recover after a restart.
- **Exactly once.** The checkpoint tracks which files are processed. Appends carry Delta's `txnAppId`/`txnVersion` (the batch id), so a batch retried after a failure is skipped. The app id changes with each checkpoint, because batch ids restart at 0 and reusing the id would make Delta skip a fresh run's batches (a bug caught in testing; regression test in `tests/test_streaming.py`). The state write is a deterministic `replaceWhere` overwrite.
- **Research in the loop.** Each car's row carries the RQ1 posterior wear for its circuit and compound (time lost vs new tyres). It also gets an **undercut window** flag in exactly the situation RQ2 studied: within 3 s of the car ahead, both on tyres 5+ laps old, green flag, 8+ laps left, at a high- or mid-wear circuit. The flag comes with RQ2's effect estimate for that tier.
- **Serving:** the Streamlit app reads `stream.race_state` through delta-rs and refreshes every 2 s.

## Alternatives considered
- **Kafka between the feed and Spark:** the standard pattern at scale, but a broker to run for a single feed of about 10 messages per second. The landing directory gives the same replayability for this volume. The producer interface would map onto a Kafka topic unchanged.
- **Spark reading MQTT directly:** no maintained Spark 3.5 MQTT source exists, and a custom source would be harder to test than a file hand-off.
- **Stateful streaming (`applyInPandasWithState`) instead of rebuilds:** less work per batch, but duplicates the batch logic in a second, stateful form. Rebuilding the state from the log keeps one implementation.
- **Polling the REST API with a token:** simpler, but rate-limited and adds seconds of latency per endpoint.

## Consequences
- The pipeline is fully exercised without a subscription: `make stream-replay` streams any race in bronze, and tests cover release timing, state logic and exactly-once behaviour end to end.
- Per-batch cost is fixed overhead, not data volume.
  - **What it is:** a micro-batch takes about 8 s whether it carries 1 event or 300 (median over three replays of the 2025 Abu Dhabi GP: [`streaming.json`](../../reports/engineering/streaming.json)). About 2 s is the Delta append and most of the rest is the state plan.
  - **What it isn't:** profiling ruled out shuffle width and adaptive execution; one-row Delta commits alone take over 1 s on Docker Desktop's disk.
  - **Measured latency:** at 60× replay speed (a whole race in about 90 s), events wait for the previous batch, giving 17–20 s median and about 25 s p95 from emitted to state written.
  - **Real-time estimate (not measured over a full race):** at real pace there is little queueing, so latency should be about one batch plus one trigger, around 10 s.
  - That is enough for lap-by-lap strategy calls (laps take about 90 s), not for anything faster. A lower-latency design would keep the state in memory and write Delta less often.
- The live connector follows OpenF1's documented interface but hasn't been run against a live session. It is the one untested path.
