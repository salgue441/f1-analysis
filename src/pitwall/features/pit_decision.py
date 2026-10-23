"""RQ2 feature tables for the undercut question (docs/research-plan.md#rq2).

Both designs start from the same pairs: at the end of lap L, driver `d` runs directly
behind rival `r` within `max_gap_s`, both mid-stint (not on an in/out lap), green flag, at
least `MIN_TYRE_LIFE` laps on their tyres, at least `MIN_LAPS_REMAINING` to go. Covariates
are measured at the end of lap L, before the decision (SAD §9.3).

feat_pit_cycle (primary). The pair's next stops are both within `window` laps and the first
of them comes on lap L+1. Treated: d stops first (the undercut). Control: r stops first.
Same-lap stops are excluded. Outcome at e = 1 + the later stop: is d ahead?

feat_pit_decision (secondary, lap-level risk set). Every pair-lap counts. Treated: d pits
on lap L+1 while r does not. Control: neither pits. Units where r pits on L+1 are excluded.
Outcome at e = 1 + the later of the two next stops, capped at L+1+horizon. Pitting on a
given lap is rare, so this design has poor propensity overlap (research plan §6).

Shared exclusions: any SC/VSC/red-flag lap in [L+1, e] (a post-treatment shock), and either
car not classified at e. The placebo outcome is the gap gain over laps L-3..L, before any
stop, which the treatment cannot have caused.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

RACE = ["season", "event_round"]
MAX_GAP_S = 3.0
HORIZON_LAPS = 15
CYCLE_WINDOW_LAPS = 8
MIN_LAPS_REMAINING = 8
MIN_TYRE_LIFE = 5


def _laps(fact_lap: DataFrame) -> DataFrame:
    by_driver = Window.partitionBy(*RACE, "driver_id").orderBy("lap_number")
    following = by_driver.rowsBetween(Window.currentRow, Window.unboundedFollowing)
    return (
        fact_lap.filter(F.col("session_type") == "R")
        .select(
            *RACE,
            "driver_id",
            "lap_number",
            "position_end_of_lap",
            "lap_end_time_s",
            "tyre_life_laps",
            "compound",
            "is_pit_in_lap",
            "is_pit_out_lap",
            "is_neutralized",
            "track_temp_c",
        )
        .withColumn(
            "next_pit_lap",
            F.min(F.when(F.col("is_pit_in_lap"), F.col("lap_number"))).over(following),
        )
    )


def _pairs(laps: DataFrame, races: DataFrame, max_gap_s: float) -> DataFrame:
    """Eligible (d behind r) pairs at the end of lap L, with `d_*` and `r_*` columns."""

    def side(prefix: str) -> DataFrame:
        return laps.select(*[F.col(c).alias(f"{prefix}{c}") for c in laps.columns])

    d, r = side("d_"), side("r_")
    pairs = (
        d.join(
            r,
            [
                F.col("d_season") == F.col("r_season"),
                F.col("d_event_round") == F.col("r_event_round"),
                F.col("d_lap_number") == F.col("r_lap_number"),
                F.col("d_position_end_of_lap") == F.col("r_position_end_of_lap") + 1,
            ],
        )
        .withColumnRenamed("d_season", "season")
        .withColumnRenamed("d_event_round", "event_round")
        .drop("r_season", "r_event_round")
        .withColumn("gap_s", F.col("d_lap_end_time_s") - F.col("r_lap_end_time_s"))
    )
    return (
        pairs.join(races, RACE)
        .withColumn("decision_lap", F.col("d_lap_number") + 1)
        .withColumn("laps_remaining", F.col("scheduled_laps") - F.col("d_lap_number"))
        .filter(F.col("gap_s").between(0, max_gap_s))
        .filter(F.col("d_lap_number") >= 3)
        .filter(F.col("laps_remaining") >= MIN_LAPS_REMAINING)
        .filter(~F.col("d_is_neutralized") & ~F.col("r_is_neutralized"))
        .filter(~F.col("d_is_pit_in_lap") & ~F.col("r_is_pit_in_lap"))
        .filter(~F.col("d_is_pit_out_lap") & ~F.col("r_is_pit_out_lap"))
        .filter(
            (F.col("d_tyre_life_laps") >= MIN_TYRE_LIFE)
            & (F.col("r_tyre_life_laps") >= MIN_TYRE_LIFE)
        )
    )


def _with_context(
    units: DataFrame,
    laps: DataFrame,
    circuit_pit_loss: DataFrame,
    circuit_deg_tier: DataFrame | None,
) -> DataFrame:
    """Rejoin traffic, outcome at `outcome_lap`, SC exclusion, placebo, final columns."""
    # Expected rejoin traffic: cars within pit_loss behind d at lap L (it would drop behind).
    others = laps.select(
        *RACE,
        F.col("lap_number").alias("d_lap_number"),
        F.col("driver_id").alias("_o_driver"),
        F.col("lap_end_time_s").alias("_o_t"),
    )
    units = units.join(circuit_pit_loss, "circuit_id", "left")
    traffic = (
        units.select(*RACE, "d_lap_number", "d_driver_id", "d_lap_end_time_s", "pit_loss_s")
        .distinct()
        .join(others, [*RACE, "d_lap_number"])
        .filter(
            (F.col("_o_driver") != F.col("d_driver_id"))
            & (F.col("_o_t") > F.col("d_lap_end_time_s"))
            & (F.col("_o_t") <= F.col("d_lap_end_time_s") + F.col("pit_loss_s"))
        )
        .groupBy(*RACE, "d_lap_number", "d_driver_id")
        .agg(F.count(F.lit(1)).alias("rejoin_traffic_cars"))
    )
    units = units.join(traffic, [*RACE, "d_lap_number", "d_driver_id"], "left").fillna(
        0, ["rejoin_traffic_cars"]
    )

    def at(lap_col: str, driver_col: str, prefix: str, offset: int = 0) -> DataFrame:
        return laps.select(
            *RACE,
            F.col("driver_id").alias(driver_col),
            (F.col("lap_number") + offset).alias(lap_col),
            F.col("position_end_of_lap").alias(f"{prefix}_pos"),
            F.col("lap_end_time_s").alias(f"{prefix}_t"),
        )

    # Both cars must be classified at e (a retirement or a timing gap drops the unit).
    units = (
        units.join(at("outcome_lap", "d_driver_id", "_d_e"), [*RACE, "d_driver_id", "outcome_lap"])
        .join(at("outcome_lap", "r_driver_id", "_r_e"), [*RACE, "r_driver_id", "outcome_lap"])
        .filter(F.col("_d_e_pos").isNotNull() & F.col("_r_e_pos").isNotNull())
    )

    # Exclude cycles disrupted by any neutralized lap in [L+1, e].
    neutral = (
        laps.filter(F.col("is_neutralized"))
        .select(*RACE, F.col("lap_number").alias("_n_lap"))
        .distinct()
    )
    disrupted = (
        units.select(*RACE, "decision_lap", "outcome_lap")
        .distinct()
        .join(neutral, RACE)
        .filter(F.col("_n_lap").between(F.col("decision_lap"), F.col("outcome_lap")))
        .select(*RACE, "decision_lap", "outcome_lap")
        .distinct()
    )
    units = units.join(disrupted, [*RACE, "decision_lap", "outcome_lap"], "left_anti")

    # Placebo: the gap three laps before L (lap L-3 shifted onto L).
    units = units.join(
        at("d_lap_number", "d_driver_id", "_d_b", offset=3),
        [*RACE, "d_lap_number", "d_driver_id"],
        "left",
    ).join(
        at("d_lap_number", "r_driver_id", "_r_b", offset=3),
        [*RACE, "d_lap_number", "r_driver_id"],
        "left",
    )
    if circuit_deg_tier is not None:
        units = units.join(circuit_deg_tier, "circuit_id", "left")
    else:
        units = units.withColumn("deg_tier", F.lit(None).cast("string"))

    gap_e = F.col("_d_e_t") - F.col("_r_e_t")
    return units.select(
        *RACE,
        F.format_string("%d_%02d", "season", "event_round").alias("race_id"),
        "circuit_id",
        F.col("d_driver_id").alias("driver_id"),
        F.col("r_driver_id").alias("rival_id"),
        "decision_lap",
        "outcome_lap",
        "treated",
        (F.col("_d_e_pos") < F.col("_r_e_pos")).cast("int").alias("outcome_ahead_after_cycle"),
        (F.col("gap_s") - gap_e).alias("outcome_gap_gain_s"),
        ((F.col("_d_b_t") - F.col("_r_b_t")) - F.col("gap_s")).alias("placebo_gap_gain_before_s"),
        "gap_s",
        F.col("d_position_end_of_lap").alias("position"),
        F.col("d_tyre_life_laps").alias("tyre_life"),
        F.col("r_tyre_life_laps").alias("rival_tyre_life"),
        F.col("d_compound").alias("compound"),
        F.col("r_compound").alias("rival_compound"),
        "laps_remaining",
        (F.col("laps_remaining") / F.col("scheduled_laps")).alias("race_fraction_remaining"),
        F.col("d_track_temp_c").alias("track_temp_c"),
        "pit_loss_s",
        "rejoin_traffic_cars",
        "deg_tier",
    )


def _races(dim_session: DataFrame) -> DataFrame:
    return dim_session.filter(F.col("session_type") == "R").select(
        "season", F.col("round").alias("event_round"), "circuit_id", "scheduled_laps"
    )


def build_feat_pit_cycle(
    fact_lap: DataFrame,
    dim_session: DataFrame,
    circuit_pit_loss: DataFrame,
    circuit_deg_tier: DataFrame | None = None,
    max_gap_s: float = MAX_GAP_S,
    window: int = CYCLE_WINDOW_LAPS,
) -> DataFrame:
    """Primary design: who stops first in a shared pit cycle (one unit per pair and cycle)."""
    laps = _laps(fact_lap)
    nd, nr = F.col("d_next_pit_lap"), F.col("r_next_pit_lap")
    units = (
        _pairs(laps, _races(dim_session), max_gap_s)
        .filter(nd.isNotNull() & nr.isNotNull())
        .filter(F.least(nd, nr) == F.col("decision_lap"))
        .filter((nd != nr) & (F.abs(nd - nr) <= window))
        .withColumn("treated", (nd < nr).cast("int"))
        .withColumn("outcome_lap", F.least(F.greatest(nd, nr) + 1, F.col("scheduled_laps")))
    )
    return _with_context(units, laps, circuit_pit_loss, circuit_deg_tier)


def build_feat_pit_decision(
    fact_lap: DataFrame,
    dim_session: DataFrame,
    circuit_pit_loss: DataFrame,
    circuit_deg_tier: DataFrame | None = None,
    max_gap_s: float = MAX_GAP_S,
    horizon: int = HORIZON_LAPS,
) -> DataFrame:
    """Secondary design: lap-level risk set (pit now vs not yet)."""
    laps = _laps(fact_lap)
    # NULL next pit (never stops again) must mean "doesn't pit", not drop the unit.
    d_pits = F.coalesce(F.col("d_next_pit_lap") == F.col("decision_lap"), F.lit(False))
    r_pits = F.coalesce(F.col("r_next_pit_lap") == F.col("decision_lap"), F.lit(False))
    units = (
        _pairs(laps, _races(dim_session), max_gap_s)
        .filter(~r_pits)
        .withColumn("treated", d_pits.cast("int"))
        # greatest/least skip NULLs: a car that never pits again doesn't extend the cycle,
        # and the horizon always caps it.
        .withColumn(
            "outcome_lap",
            F.least(
                F.greatest(F.col("d_next_pit_lap"), F.col("r_next_pit_lap")) + 1,
                F.col("decision_lap") + horizon,
                F.col("scheduled_laps"),
            ),
        )
    )
    return _with_context(units, laps, circuit_pit_loss, circuit_deg_tier)


def circuit_pit_loss(fact_pit_stop: DataFrame, dim_session: DataFrame) -> DataFrame:
    """Median green-flag pit-lane time per circuit: a static circuit attribute."""
    races = dim_session.filter(F.col("session_type") == "R").select("season", "round", "circuit_id")
    return (
        fact_pit_stop.filter(~F.col("under_neutralization") & F.col("pit_lane_s").between(10, 60))
        .join(races, ["season", "round"])
        .groupBy("circuit_id")
        .agg(F.percentile_approx("pit_lane_s", 0.5).alias("pit_loss_s"))
    )
