"""feat_pit_decision (RQ2): driver x rival-ahead x lap units for the undercut question.

Unit: at the end of lap t-1, driver `d` runs directly behind rival `r` within `max_gap_s`.
The decision is whether `d` pits on lap t (its in-lap is t).

- treated: d pits on lap t and r does not.
- control: neither pits on lap t.
- excluded: r pits on lap t (the rival moved first, or both did).

Covariates are all measured at the end of lap t-1, i.e. before the decision (SAD §9.3).

Outcome: positions once both cars have pitted, at lap e = 1 + max(next pit of d, next pit of
r), capped at t + horizon and the last lap. `outcome_ahead_after_cycle` asks whether d is
ahead of r at e. `outcome_gap_gain_s` is the gap at t-1 minus the gap at e (positive means
d gained). Cycles with any SC/VSC/red-flag lap in [t, e] are excluded: the neutralization
is a post-treatment shock.

Placebo outcome: the gap gain over laps t-4 .. t-1, before any stop, which the treatment
cannot have caused.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

RACE = ["season", "event_round"]
MAX_GAP_S = 3.0
HORIZON_LAPS = 15
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


def build_feat_pit_decision(
    fact_lap: DataFrame,
    dim_session: DataFrame,
    circuit_pit_loss: DataFrame,
    circuit_deg_tier: DataFrame | None = None,
    max_gap_s: float = MAX_GAP_S,
    horizon: int = HORIZON_LAPS,
) -> DataFrame:
    """`circuit_pit_loss`: circuit_id, pit_loss_s. `circuit_deg_tier`: circuit_id, deg_tier."""
    laps = _laps(fact_lap)
    races = dim_session.filter(F.col("session_type") == "R").select(
        "season", F.col("round").alias("event_round"), "circuit_id", "scheduled_laps"
    )

    def side(prefix: str) -> DataFrame:
        return laps.select(*[F.col(c).alias(f"{prefix}{c}") for c in laps.columns])

    d, r = side("d_"), side("r_")
    # Pairs at the end of lap t-1: r is one place ahead of d.
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
    ).withColumn("gap_s", F.col("d_lap_end_time_s") - F.col("r_lap_end_time_s"))

    units = (
        pairs.join(races, RACE)
        .withColumn("decision_lap", F.col("d_lap_number") + 1)
        .withColumn("laps_remaining", F.col("scheduled_laps") - F.col("d_lap_number"))
        .filter(F.col("gap_s").between(0, max_gap_s))
        .filter(F.col("d_lap_number") >= 3)
        .filter(F.col("laps_remaining") >= MIN_LAPS_REMAINING)
        .filter(~F.col("d_is_neutralized") & ~F.col("r_is_neutralized"))
        # Neither car is on its in- or out-lap at t-1: both are mid-stint.
        .filter(~F.col("d_is_pit_in_lap") & ~F.col("r_is_pit_in_lap"))
        .filter(~F.col("d_is_pit_out_lap") & ~F.col("r_is_pit_out_lap"))
        .filter(
            (F.col("d_tyre_life_laps") >= MIN_TYRE_LIFE)
            & (F.col("r_tyre_life_laps") >= MIN_TYRE_LIFE)
        )
    )

    # Treatment at lap t, read from each driver's own next pit lap (as of lap t-1).
    # NULL next pit (never stops again) must mean "doesn't pit", not drop the unit.
    d_pits = F.coalesce(F.col("d_next_pit_lap") == F.col("decision_lap"), F.lit(False))
    r_pits = F.coalesce(F.col("r_next_pit_lap") == F.col("decision_lap"), F.lit(False))
    units = (
        units.filter(~r_pits)
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

    # Expected rejoin traffic: cars within pit_loss behind d at t-1 (it would drop behind them).
    others = laps.select(
        *RACE,
        F.col("lap_number").alias("d_lap_number"),
        F.col("driver_id").alias("_o_driver"),
        F.col("lap_end_time_s").alias("_o_t"),
    )
    units = units.join(circuit_pit_loss, "circuit_id", "left")
    traffic = (
        units.select(*RACE, "d_lap_number", "d_driver_id", "d_lap_end_time_s", "pit_loss_s")
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

    # Outcome positions and gap at lap e.
    at_e_d = laps.select(
        *RACE,
        F.col("driver_id").alias("d_driver_id"),
        F.col("lap_number").alias("outcome_lap"),
        F.col("position_end_of_lap").alias("_d_pos_e"),
        F.col("lap_end_time_s").alias("_d_t_e"),
    )
    at_e_r = laps.select(
        *RACE,
        F.col("driver_id").alias("r_driver_id"),
        F.col("lap_number").alias("outcome_lap"),
        F.col("position_end_of_lap").alias("_r_pos_e"),
        F.col("lap_end_time_s").alias("_r_t_e"),
    )
    units = units.join(at_e_d, [*RACE, "d_driver_id", "outcome_lap"]).join(
        at_e_r, [*RACE, "r_driver_id", "outcome_lap"]
    )

    # Exclude cycles disrupted by any neutralized lap in [t, e].
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

    # Placebo: gap 3 laps before t-1.
    before = laps.select(
        *RACE,
        (F.col("lap_number") + 3).alias("d_lap_number"),
        F.col("driver_id").alias("d_driver_id"),
        F.col("lap_end_time_s").alias("_d_t_before"),
    )
    before_r = laps.select(
        *RACE,
        (F.col("lap_number") + 3).alias("d_lap_number"),
        F.col("driver_id").alias("r_driver_id"),
        F.col("lap_end_time_s").alias("_r_t_before"),
    )
    units = units.join(before, [*RACE, "d_lap_number", "d_driver_id"], "left").join(
        before_r, [*RACE, "d_lap_number", "r_driver_id"], "left"
    )
    if circuit_deg_tier is not None:
        units = units.join(circuit_deg_tier, "circuit_id", "left")
    else:
        units = units.withColumn("deg_tier", F.lit(None).cast("string"))

    gap_e = F.col("_d_t_e") - F.col("_r_t_e")
    return units.select(
        *RACE,
        F.format_string("%d_%02d", "season", "event_round").alias("race_id"),
        "circuit_id",
        F.col("d_driver_id").alias("driver_id"),
        F.col("r_driver_id").alias("rival_id"),
        "decision_lap",
        "outcome_lap",
        "treated",
        (F.col("_d_pos_e") < F.col("_r_pos_e")).cast("int").alias("outcome_ahead_after_cycle"),
        (F.col("gap_s") - gap_e).alias("outcome_gap_gain_s"),
        ((F.col("_d_t_before") - F.col("_r_t_before")) - F.col("gap_s")).alias(
            "placebo_gap_gain_before_s"
        ),
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


def circuit_pit_loss(fact_pit_stop: DataFrame, dim_session: DataFrame) -> DataFrame:
    """Median green-flag pit-lane time per circuit: a static circuit attribute."""
    races = dim_session.filter(F.col("session_type") == "R").select("season", "round", "circuit_id")
    return (
        fact_pit_stop.filter(~F.col("under_neutralization") & F.col("pit_lane_s").between(10, 60))
        .join(races, ["season", "round"])
        .groupBy("circuit_id")
        .agg(F.percentile_approx("pit_lane_s", 0.5).alias("pit_loss_s"))
    )
