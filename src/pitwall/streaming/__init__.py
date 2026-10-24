"""Race-day streaming (ADR-0007): OpenF1 events -> landing files -> Structured Streaming.

Producers (`replay` for past races, `live` for OpenF1's MQTT feed) write the same JSON-lines
files into a landing directory; `pipeline` consumes them with Spark Structured Streaming and
keeps `stream.events` (the raw log) and `stream.race_state` (one row per car) up to date.
"""
