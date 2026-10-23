import numpy as np
import pandas as pd
import pytest

from pitwall.features.pit_decision import build_feat_pit_decision
from pitwall.models import causal

LAP_SCHEMA = (
    "season int, event_round int, session_type string, driver_id string, lap_number int, "
    "position_end_of_lap int, lap_end_time_s double, tyre_life_laps int, compound string, "
    "is_pit_in_lap boolean, is_pit_out_lap boolean, is_neutralized boolean, "
    "track_temp_c double"
)


def _race_laps(d_pit, r_pit, laps=30, sc_lap=None, swap_after=None):
    """Two cars, 'a' ahead of 'b' by 1.0 s. Optionally swap order from lap `swap_after`."""
    rows = []
    for lap in range(1, laps + 1):
        a_ahead = swap_after is None or lap < swap_after
        for driver, pit in (("a", r_pit), ("b", d_pit)):
            first = (driver == "a") == a_ahead
            pos = 1 if first else 2
            t = 90.0 * lap + (0.0 if first else 1.0)
            age = lap if pit is None or lap <= pit else lap - pit
            rows.append(
                (
                    2024,
                    1,
                    "R",
                    driver,
                    lap,
                    pos,
                    t,
                    age,
                    "MEDIUM",
                    lap == pit,
                    pit is not None and lap == pit + 1,
                    lap == sc_lap,
                    30.0,
                )
            )
    return rows


def _build(spark, rows):
    laps = spark.createDataFrame(rows, LAP_SCHEMA)
    sessions = spark.createDataFrame(
        [(2024, 1, "R", "bahrain", 30)],
        "season int, round int, session_type string, circuit_id string, scheduled_laps int",
    )
    pit_loss = spark.createDataFrame([("bahrain", 22.0)], "circuit_id string, pit_loss_s double")
    return build_feat_pit_decision(laps, sessions, pit_loss)


def test_units_treatment_and_outcome(spark):
    # b (behind) pits on lap 12, a on lap 14; b comes out ahead from lap 15.
    all_units = _build(spark, _race_laps(d_pit=12, r_pit=14, swap_after=15)).toPandas()
    # After the swap a runs behind b; those are separate units with a as the driver.
    assert set(zip(all_units["driver_id"], all_units["rival_id"], strict=True)) == {
        ("b", "a"),
        ("a", "b"),
    }
    units = all_units[all_units["driver_id"] == "b"]
    treated = units[units["treated"] == 1]
    assert list(treated["decision_lap"]) == [12]
    row = treated.iloc[0]
    assert row["outcome_lap"] == 15 and row["outcome_ahead_after_cycle"] == 1
    assert row["gap_s"] == pytest.approx(1.0) and row["outcome_gap_gain_s"] == pytest.approx(2.0)
    # Controls only before the stop (tyres >= 5 laps old from lap 5, so decisions 6..11).
    assert sorted(units.loc[units["treated"] == 0, "decision_lap"]) == list(range(6, 12))
    assert (units["outcome_lap"] == 15).all()


def test_sc_disrupted_cycles_are_excluded(spark):
    units = _build(spark, _race_laps(d_pit=12, r_pit=14, sc_lap=13)).toPandas()
    assert 12 not in set(units["decision_lap"])


def _confounded(n_races=60, per_race=40, effect=0.15, seed=0):
    """Older tyres make pitting likelier AND make being ahead later likelier for other reasons."""
    rng = np.random.default_rng(seed)
    rows = []
    for race in range(n_races):
        race_shift = rng.normal(0, 0.3)
        for _ in range(per_race):
            tyre = rng.uniform(5, 35)
            gap = rng.uniform(0, 3)
            p = 1 / (1 + np.exp(-(-3 + 0.12 * tyre - 0.4 * gap)))
            t = rng.random() < p
            logit = -0.5 + 0.1 * (tyre - 20) - 0.5 * gap + race_shift
            y = rng.random() < 1 / (1 + np.exp(-logit)) + effect * t
            rows.append(
                {
                    "race_id": f"r{race}",
                    "treated": int(t),
                    "outcome_ahead_after_cycle": int(y),
                    "gap_s": gap,
                    "position": rng.integers(2, 20),
                    "tyre_life": tyre,
                    "rival_tyre_life": tyre + rng.normal(0, 3),
                    "laps_remaining": 30,
                    "race_fraction_remaining": 0.5,
                    "track_temp_c": 35.0,
                    "pit_loss_s": 22.0,
                    "rejoin_traffic_cars": rng.integers(0, 4),
                    "compound": "MEDIUM",
                    "rival_compound": "MEDIUM",
                    "deg_tier": "mid",
                }
            )
    return pd.DataFrame(rows)


def test_estimators_remove_confounding_bias():
    df = _confounded(effect=0.15)
    outcome = "outcome_ahead_after_cycle"
    naive = causal.naive(df, outcome)
    e_hat, g_hat = causal.cross_fit(df, outcome, seed=0)
    ipw = causal.ipw(df, outcome, e_hat, n_boot=200)
    dml = causal.dml(df, outcome, e_hat, g_hat)
    # Naive is biased upward by tyre age; adjusted estimates cover the true 0.15.
    assert naive.effect > 0.2
    assert dml.lo <= 0.15 <= dml.hi and ipw.lo <= 0.15 <= ipw.hi
    assert abs(dml.effect - 0.15) < abs(naive.effect - 0.15)

    before = causal.smd(df).abs()
    after = causal.smd(df, causal.ipw_weights(df["treated"].to_numpy(), e_hat)).abs()
    assert before["tyre_life"] > 0.3 and after["tyre_life"] < before["tyre_life"]


def test_e_value_and_refuter():
    assert causal.e_value(1.0) == pytest.approx(1.0)
    assert causal.e_value(2.0) == pytest.approx(2 + np.sqrt(2))
    assert causal.e_value(0.5) == pytest.approx(causal.e_value(2.0))
    df = _confounded(n_races=30, per_race=30, effect=0.2, seed=1)
    ref = causal.permuted_treatment_refuter(df, "outcome_ahead_after_cycle", n_perm=5)
    assert abs(ref["mean_effect"]) < 0.08
