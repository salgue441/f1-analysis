"""Structured Streaming: landing files -> stream.events, stream.race_state, stream._batches.

One query per session. Each micro-batch:

1. appends its records to `stream.events` (the raw log, partitioned by session);
2. rebuilds `stream.race_state` for the session from the full log (a few thousand rows per
   race, so a rebuild is cheaper and simpler than incremental state) and replaces that
   session's rows;
3. logs counts and end-to-end latency (emitted -> state written) to `stream._batches`.

Writes are idempotent: the appends carry Delta's `txnAppId`/`txnVersion`, so a micro-batch
replayed after a failure (same batch id from the checkpoint) is skipped, and the state
rebuild is a deterministic overwrite. Exactly-once end to end.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery

from pitwall.config import get_settings
from pitwall.paths import table_path
from pitwall.streaming.landing import LandingWriter
from pitwall.streaming.state import StrategyContext, build_race_state

log = logging.getLogger(__name__)

REPORTS = Path(__file__).resolve().parents[3] / "reports"
STREAM_SHUFFLE_PARTITIONS = 2


@dataclass(frozen=True)
class StreamSession:
    session_key: int
    season: int
    event_round: int
    circuit_id: str | None = None


def landing_dir(session_key: int) -> Path:
    return Path(get_settings().lakehouse_root).parent / "landing" / "openf1" / str(session_key)


def checkpoint_dir(session_key: int) -> str:
    return table_path("stream", f"_checkpoints/{session_key}")


def _exists(spark: SparkSession, path: str) -> bool:
    from delta.tables import DeltaTable

    return DeltaTable.isDeltaTable(spark, path)


def stream_app_id(session_key: int) -> str:
    """Delta idempotency id, one per checkpoint.

    Batch ids restart at 0 with a new checkpoint, so the id must change with it: reusing it
    would make Delta skip the new run's batches as already committed. It is kept next to the
    checkpoint and removed with it.
    """
    path = Path(checkpoint_dir(session_key) + ".appid")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"pitwall-stream-{session_key}-{uuid.uuid4().hex[:12]}")
    return path.read_text().strip()


EVENT_COLUMNS = ["session_key", "endpoint", "payload"]


class BatchProcessor:
    """`foreachBatch` handler for one session.

    The session's event log is kept in memory (appended per batch and checkpointed locally),
    so a batch costs the same at lap 50 as at lap 1. Delta is the durable copy, read once to
    recover after a restart.
    """

    def __init__(self, session_key: int, ctx: StrategyContext):
        self.session_key = session_key
        self.ctx = ctx
        self.app_id = stream_app_id(session_key)
        self._log: DataFrame | None = None

    def _session_log(self, spark: SparkSession, new: DataFrame) -> DataFrame:
        if self._log is None:  # first batch of this process: recover from the durable log
            log_df = (
                spark.read.format("delta")
                .load(table_path("stream", "events"))
                .filter(F.col("session_key") == self.session_key)
                .select(*EVENT_COLUMNS)
            )
        else:
            log_df = self._log.unionByName(new.select(*EVENT_COLUMNS))
        old, self._log = self._log, log_df.localCheckpoint(eager=True)
        if old is not None:
            old.unpersist()
        return self._log

    def _append(self, df: DataFrame, table: str, batch_id: int) -> None:
        (
            df.write.format("delta")
            .mode("append")
            .partitionBy("session_key")
            .option("txnAppId", self.app_id)
            .option("txnVersion", batch_id)
            .option("mergeSchema", "true")
            .save(table_path("stream", table))
        )

    def __call__(self, batch: DataFrame, batch_id: int) -> None:
        started = time.perf_counter()
        spark = batch.sparkSession
        endpoint = F.get_json_object("value", "$._endpoint")
        record_key = F.get_json_object("value", "$.session_key").cast("long")
        events = (
            batch.select(
                F.coalesce(record_key, F.lit(self.session_key)).alias("session_key"),
                endpoint.alias("endpoint"),
                F.col("value").alias("payload"),
                F.to_timestamp(F.get_json_object("value", "$._emitted_at")).alias("emitted_at"),
                F.lit(batch_id).cast("long").alias("stream_batch_id"),
            )
            .filter(F.col("session_key") == self.session_key)
            .filter(endpoint.isNotNull())
            .cache()
        )
        stats = events.agg(
            F.count("*").alias("n"),
            F.min("emitted_at").alias("first_emitted"),
            F.max("emitted_at").alias("last_emitted"),
        ).first()
        if not stats or not stats["n"]:
            events.unpersist()
            return
        self._append(events, "events", batch_id)
        appended = time.perf_counter()
        session_log = self._session_log(spark, events)
        events.unpersist()

        state = build_race_state(session_log, self.ctx)
        state_path = table_path("stream", "race_state")
        writer = state.write.format("delta").partitionBy("session_key")
        if _exists(spark, state_path):
            writer = writer.mode("overwrite").option(
                "replaceWhere", f"session_key = {self.session_key}"
            )
        writer.save(state_path)

        done = spark.sql("select current_timestamp() as t").first()["t"]  # type: ignore[index]
        latency = (done - stats["first_emitted"]).total_seconds()
        row = {
            "session_key": self.session_key,
            "stream_batch_id": batch_id,
            "n_events": int(stats["n"]),
            "first_emitted_at": stats["first_emitted"],
            "last_emitted_at": stats["last_emitted"],
            "state_written_at": done,
            "max_latency_s": latency,
            "min_latency_s": (done - stats["last_emitted"]).total_seconds(),
            "append_s": appended - started,
            "processing_s": time.perf_counter() - started,
        }
        self._append(spark.createDataFrame([row]), "_batches", batch_id)
        log.info(
            "stream batch %s: %s events in %.1fs, latency <= %.1fs",
            batch_id,
            stats["n"],
            row["processing_s"],
            latency,
        )


def start_stream(
    spark: SparkSession,
    session: StreamSession,
    ctx: StrategyContext,
    trigger_s: float = 2.0,
    available_now: bool = False,
) -> StreamingQuery:
    landing = landing_dir(session.session_key)
    landing.mkdir(parents=True, exist_ok=True)
    # A race's log is a few thousand rows: batch ETL's shuffle width only adds task overhead.
    spark.conf.set("spark.sql.shuffle.partitions", str(STREAM_SHUFFLE_PARTITIONS))
    writer = (
        spark.readStream.format("text")
        .load(str(landing))
        .writeStream.foreachBatch(BatchProcessor(session.session_key, ctx))
        .option("checkpointLocation", checkpoint_dir(session.session_key))
        .queryName(f"race_state_{session.session_key}")
    )
    trigger = (
        {"availableNow": True} if available_now else {"processingTime": f"{trigger_s} seconds"}
    )
    return writer.trigger(**trigger).start()


# --- inputs from the lakehouse ----------------------------------------------------------------


def _read(layer: str, table: str) -> DataFrame:
    from pitwall.spark import get_spark

    return get_spark().read.format("delta").load(table_path(layer, table))  # type: ignore[arg-type]


def strategy_context(session: StreamSession) -> StrategyContext:
    """Scheduled laps (dim_session), RQ1 wear for the circuit, RQ2 effect for its tier."""
    races = _read("gold", "dim_session").filter(
        (F.col("season") == session.season)
        & (F.col("round") == session.event_round)
        & (F.col("session_type") == "R")
    )
    race = races.select("scheduled_laps", "circuit_id").first()
    circuit = session.circuit_id or (race["circuit_id"] if race else None)
    laps = int(race["scheduled_laps"]) if race and race["scheduled_laps"] else 60

    from pitwall.spark import get_spark

    deg: dict[str, float] = {}
    tier = None
    posterior_path = table_path("features", "rq1_degradation_posterior")
    if circuit and _exists(get_spark(), posterior_path):
        post = _read("features", "rq1_degradation_posterior")
        rows = (
            post.filter(F.col("circuit_id") == circuit)
            .groupBy("compound")
            .agg(F.avg("beta_s_per_lap").alias("b"))
            .collect()
        )
        deg = {r["compound"]: float(r["b"]) for r in rows}
        from pitwall.research import circuit_deg_tiers

        tiers = circuit_deg_tiers().set_index("circuit_id")["deg_tier"]
        tier = tiers.get(circuit)

    effect = None
    rq2 = REPORTS / "rq2" / "results.json"
    if tier and rq2.exists():
        by_tier = json.loads(rq2.read_text())["cycle"]["by_deg_tier"]
        effect = 100 * by_tier[tier]["effect"] if tier in by_tier else None
    return StrategyContext(laps, deg, tier, effect)


def replay_payloads(season: int, event_round: int) -> tuple[int, dict[str, list[dict[str, Any]]]]:
    """(session_key, records per endpoint) for a race from bronze (latest batch of each)."""
    from pitwall.streaming.landing import STREAM_ENDPOINTS

    payloads: dict[str, list[dict[str, Any]]] = {}
    session_key = None
    for endpoint in STREAM_ENDPOINTS:
        rows = (
            _read("bronze", f"openf1_{endpoint}")
            .filter(
                (F.col("season") == season)
                & (F.col("event_round") == event_round)
                & (F.col("session_type") == "R")
            )
            .orderBy(F.col("_ingested_at").desc())
            .select("session_key", "payload")
            .limit(1)
            .collect()
        )
        if rows:
            session_key = int(rows[0]["session_key"])
            payloads[endpoint] = json.loads(rows[0]["payload"])
    if session_key is None:
        raise ValueError(f"No OpenF1 race in bronze for {season} round {event_round}")
    return session_key, payloads


def reset_session(session_key: int) -> None:
    """Remove a session's landing files and checkpoint so a replay starts from scratch."""
    import shutil

    from pitwall.spark import get_spark

    shutil.rmtree(landing_dir(session_key), ignore_errors=True)
    shutil.rmtree(checkpoint_dir(session_key), ignore_errors=True)
    Path(checkpoint_dir(session_key) + ".appid").unlink(missing_ok=True)
    spark = get_spark()
    for table in ("events", "race_state", "_batches"):
        path = table_path("stream", table)
        if _exists(spark, path):
            from delta.tables import DeltaTable

            DeltaTable.forPath(spark, path).delete(F.col("session_key") == session_key)


def run_replay(
    season: int,
    event_round: int,
    speed: float = 30.0,
    flush_every_s: float = 1.0,
    trigger_s: float = 2.0,
) -> dict[str, Any]:
    """Replay a race through the streaming pipeline and wait until it has caught up."""
    from pitwall.spark import get_spark
    from pitwall.streaming.replay import release_times, replay

    spark = get_spark("pitwall-stream")
    session_key, payloads = replay_payloads(season, event_round)
    session = StreamSession(session_key, season, event_round)
    reset_session(session_key)
    ctx = strategy_context(session)
    events = release_times(payloads)
    writer = LandingWriter(landing_dir(session_key))
    query = start_stream(spark, session, ctx, trigger_s)
    producer = threading.Thread(
        target=replay,
        args=(events, writer),
        kwargs={"speed": speed, "flush_every_s": flush_every_s},
    )
    producer.start()
    try:
        producer.join()
        query.processAllAvailable()
    finally:
        query.stop()
    summary = {
        "season": season,
        "round": event_round,
        "speed": speed,
        "trigger_s": trigger_s,
        "lakehouse_root": get_settings().lakehouse_root,
        **_summary(session_key, writer, ctx),
    }
    out = REPORTS / "engineering" / "streaming.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def run_live_stream(
    session: StreamSession, username: str, password: str, trigger_s: float = 2.0
) -> dict[str, Any]:
    """Consume OpenF1's live feed for `session` until interrupted (Ctrl-C)."""
    from pitwall.spark import get_spark
    from pitwall.streaming.live import run_live

    spark = get_spark("pitwall-stream")
    ctx = strategy_context(session)
    writer = LandingWriter(landing_dir(session.session_key))
    query = start_stream(spark, session, ctx, trigger_s)
    stopping = threading.Event()
    producer = threading.Thread(
        target=run_live,
        args=(username, password, writer, session.session_key),
        kwargs={"stop": stopping.is_set},
    )
    producer.start()
    try:
        while producer.is_alive():
            producer.join(timeout=1.0)
    except KeyboardInterrupt:
        log.info("Stopping the live feed")
    finally:
        stopping.set()
        producer.join()
        query.processAllAvailable()
        query.stop()
    return _summary(session.session_key, writer, ctx)


def _summary(session_key: int, writer: LandingWriter, ctx: StrategyContext) -> dict[str, Any]:
    batches = _read("stream", "_batches").filter(F.col("session_key") == session_key).toPandas()
    return {
        "session_key": session_key,
        "events": writer.records_written,
        "landing_files": writer.files_written,
        "micro_batches": len(batches),
        "latency_s_median": float(batches["max_latency_s"].median()),
        "latency_s_p95": float(batches["max_latency_s"].quantile(0.95)),
        "batch_processing_s_median": float(batches["processing_s"].median()),
        "batch_processing_s_p95": float(batches["processing_s"].quantile(0.95)),
        "delta_append_s_median": float(batches["append_s"].median()),
        "context": ctx.__dict__,
    }
