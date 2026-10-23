"""stream.race_state: one row per car, rebuilt from every event received so far.

Pure DataFrame -> DataFrame like the batch layers, and it reuses their pieces: the OpenF1
record schemas and track-status coding from `silver.openf1`. Strategy context comes from
the research results:

- `time_loss_vs_new_s`: RQ1 posterior mean wear for this circuit and compound x tyre age;
- `undercut_window`: the car is in the situation RQ2 studied (within 3 s of the car ahead,
  both on tyres at least 5 laps old, green flag, at least 8 laps left) at a circuit where
  RQ2 found the undercut works (high or mid degradation tier), with RQ2's effect for it.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from pitwall.silver.fastf1 import TRACK_STATUS_LABELS
from pitwall.silver.openf1 import SCHEMAS, status_events

UNDERCUT_GAP_S = 3.0
MIN_TYRE_AGE = 5
MIN_LAPS_LEFT = 8
UNDERCUT_TIERS = ("high", "mid")


@dataclass(frozen=True)
class StrategyContext:
    """Per-race constants: scheduled laps, RQ1 wear by compound, RQ2 effect for the tier."""

    race_laps: int
    deg_s_per_lap: dict[str, float]
    tier: str | None = None
    undercut_effect_pp: float | None = None


def parse(events: DataFrame, endpoint: str) -> DataFrame:
    fields = ", ".join(
        f"{n}:{t}" for n, t in (c.strip().split(" ") for c in SCHEMAS[endpoint].split(","))
    )
    return (
        events.filter(F.col("endpoint") == endpoint)
        .select("session_key", F.from_json("payload", f"struct<{fields}>").alias("r"))
        .select("session_key", "r.*")
    )


def _latest(df: DataFrame, keys: list[str], order: Column) -> DataFrame:
    w = Window.partitionBy(*keys).orderBy(order.desc())
    return df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")


def _epoch(col: str) -> Column:
    return F.to_timestamp(col).cast("double")


def build_race_state(events: DataFrame, ctx: StrategyContext) -> DataFrame:
    """`events`: session_key, endpoint, payload (JSON record). Output grain: session x car."""
    car = ["session_key", "driver_number"]
    laps = (
        parse(events, "laps")
        .withColumn("lap_end", _epoch("date_start") + F.col("lap_duration"))
        .filter(F.col("lap_end").isNotNull())
    )
    last_lap = _latest(laps, car, F.col("lap_number")).select(
        *car, "lap_number", F.col("lap_duration").alias("last_lap_s"), "lap_end"
    )
    position = _latest(
        parse(events, "position").withColumn("_t", _epoch("date")), car, F.col("_t")
    ).select(*car, "position")
    stint = _latest(parse(events, "stints"), car, F.col("stint_number")).select(
        *car, "stint_number", "compound", "lap_start", "tyre_age_at_start"
    )
    stops = parse(events, "pit").groupBy(*car).agg(F.count("*").alias("pit_stops"))
    drivers = _latest(parse(events, "drivers"), car, F.lit(0)).select(
        *car, F.col("name_acronym").alias("driver"), F.col("team_name").alias("team")
    )

    # Track status: the latest race-control message that changes it (silver's coding).
    rc = parse(events, "race_control").select(
        F.lit(0).alias("season"),
        F.lit(0).alias("event_round"),
        F.lit("R").alias("session_type"),
        F.lit(0).cast("timestamp").alias("session_start_utc"),
        F.lit(None).cast("timestamp").alias("_ingested_at"),
        F.lit(None).cast("string").alias("_source_batch_id"),
        "session_key",
        "date",
        "category",
        "flag",
        "scope",
        "message",
    )
    status = status_events(rc.drop("session_key")).select("start_s", "status")
    labels = F.create_map(*[F.lit(x) for kv in TRACK_STATUS_LABELS.items() for x in kv])
    current = (
        status.orderBy(F.col("start_s").desc())
        .limit(1)
        .select(labels[F.col("status")].alias("track_status"))
        .collect()
    )
    track_status = F.lit(current[0]["track_status"] if current else "green")

    state = (
        position.join(drivers, car, "left")
        .join(last_lap, car, "left")
        .join(stint, car, "left")
        .join(stops, car, "left")
        .fillna(0, subset=["pit_stops"])
        .withColumn(
            "tyre_age_laps",
            F.col("tyre_age_at_start")
            + F.greatest(F.col("lap_number") - F.col("lap_start") + 1, F.lit(0)),
        )
    )

    # Gap at the line to the car ahead, on the last lap both have completed.
    ahead = state.select(
        "session_key",
        (F.col("position") + 1).alias("position"),
        F.col("driver_number").alias("ahead_number"),
        F.col("tyre_age_laps").alias("ahead_tyre_age"),
    )
    ahead_laps = laps.select(
        "session_key",
        F.col("driver_number").alias("ahead_number"),
        "lap_number",
        F.col("lap_end").alias("ahead_lap_end"),
    )
    leader_lap = F.max("lap_number").over(Window.partitionBy("session_key"))
    # A circuit without an RQ1 posterior (a new venue) has no wear estimate.
    deg_rate = (
        F.create_map(*[F.lit(x) for kv in ctx.deg_s_per_lap.items() for x in kv])[F.col("compound")]
        if ctx.deg_s_per_lap
        else F.lit(None).cast("double")
    )
    in_tier = F.lit(ctx.tier in UNDERCUT_TIERS)
    window_open = (
        (track_status == "green")
        & F.col("gap_ahead_s").between(0, UNDERCUT_GAP_S)
        & (F.col("tyre_age_laps") >= MIN_TYRE_AGE)
        & (F.col("ahead_tyre_age") >= MIN_TYRE_AGE)
        & (F.col("laps_remaining") >= MIN_LAPS_LEFT)
        & in_tier
    )
    return (
        state.join(ahead, ["session_key", "position"], "left")
        .join(ahead_laps, ["session_key", "ahead_number", "lap_number"], "left")
        .withColumn("gap_ahead_s", F.round(F.col("lap_end") - F.col("ahead_lap_end"), 3))
        .withColumn("laps_remaining", F.lit(ctx.race_laps) - leader_lap)
        .withColumn("track_status", track_status)
        .withColumn("time_loss_vs_new_s", F.round(deg_rate * F.col("tyre_age_laps"), 3))
        .withColumn("undercut_window", F.coalesce(window_open, F.lit(False)))
        .withColumn(
            "undercut_effect_pp",
            F.when(F.col("undercut_window"), F.lit(ctx.undercut_effect_pp).cast("double")),
        )
        .withColumn("last_lap_end_utc", F.col("lap_end").cast("timestamp"))
        .withColumn("_updated_at", F.current_timestamp())
        .select(
            "session_key",
            "driver_number",
            "driver",
            "team",
            "position",
            "lap_number",
            "last_lap_s",
            "gap_ahead_s",
            "compound",
            "stint_number",
            "tyre_age_laps",
            "pit_stops",
            "time_loss_vs_new_s",
            "track_status",
            "laps_remaining",
            "undercut_window",
            "undercut_effect_pp",
            "last_lap_end_utc",
            "_updated_at",
        )
        .orderBy("position")
    )
