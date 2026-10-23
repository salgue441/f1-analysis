# ADR-0002: Medallion layers with a Kimball star schema in gold

- **Status:** Accepted
- **Date:** 2026-10-22

## Context
Three sources describe overlapping entities with different identifiers and quality. Analysts (and models) need one consistent, documented view, and raw data must stay replayable.

## Decision
- **Bronze:** append-only raw data + ingestion metadata.
- **Silver:** typed, deduplicated, conformed entities; MERGE on natural keys; quality gates with quarantine.
- **Gold:** Kimball star schema (`fact_lap` at the center) with **deterministic hash surrogate keys**. Driver–team history is **SCD2**.
- **Features:** separate per-research-question tables derived from gold, point-in-time correct.

## Alternatives considered
- **One big wide table:** easy to start, but duplicates logic across RQs and has no clear grain.
- **Data Vault:** strong for auditability with many sources, but overkill for three sources and one consumer.
- **Sequence/identity surrogate keys:** non-deterministic across rebuilds, which breaks idempotency.

## Consequences
- More tables to maintain, but each has one documented grain and owner.
- Features are reproducible from gold, and gold from bronze.
