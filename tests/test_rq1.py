import numpy as np
import pandas as pd
import pytest

from pitwall.features.tire_degradation import build_feat_tire_degradation
from pitwall.models import evaluation
from pitwall.models.baselines import B0CompoundSlope, stint_slopes
from pitwall.models.synthetic import simulate_laps
from pitwall.models.tire_degradation import (
    TempScaler,
    degradation_table,
    hypothesis_tests,
    predictive_draws,
    shape_draws,
)

FACT_LAP_SCHEMA = (
    "season int, event_round int, session_type string, driver_id string, "
    "constructor_id string, lap_number int, stint_number int, compound string, "
    "tyre_life_laps int, lap_time_s double, gap_to_car_ahead_s double, track_temp_c double, "
    "is_green_flag boolean, is_accurate boolean, is_pit_in_lap boolean, "
    "is_pit_out_lap boolean, is_deleted boolean"
)


def _fl(lap, time=95.0, compound="SOFT", green=True, pit_in=False, gap=3.0, temp=30.0):
    return (
        2024,
        1,
        "R",
        "ver",
        "rbr",
        lap,
        1,
        compound,
        lap,
        time,
        gap,
        temp,
        green,
        True,
        pit_in,
        False,
        False,
    )


def _dims(spark):
    session = spark.createDataFrame(
        [(2024, 1, "R", "bahrain", 57)],
        "season int, round int, session_type string, circuit_id string, scheduled_laps int",
    )
    results = spark.createDataFrame(
        [(2024, 1, "Q", 89.0)], "season int, round int, session_type string, best_lap_time_s double"
    )
    return session, results


def test_feature_filters_and_columns(spark):
    laps = spark.createDataFrame(
        [
            _fl(1),  # lap 1: excluded
            _fl(2, gap=1.0),
            _fl(3, green=False),  # SC lap
            _fl(4, pit_in=True),
            _fl(5, compound="INTERMEDIATE"),
            _fl(6, time=120.0),  # > 1.15 x pole
            _fl(7, gap=None),  # leader
        ],
        FACT_LAP_SCHEMA,
    )
    rows = {r.lap_number: r for r in build_feat_tire_degradation(laps, *_dims(spark)).collect()}
    assert set(rows) == {2, 7}
    assert rows[2].fuel_proxy == 55 and rows[2].in_traffic is True
    assert rows[7].in_traffic is False and rows[2].race_id == "2024_01"


def test_features_are_point_in_time(spark):
    """Changing later laps must not change an earlier lap's features."""
    base = [_fl(n, time=95.0 + 0.05 * n) for n in range(2, 10)]
    changed = base[:4] + [_fl(n, time=99.0, gap=0.5, temp=50.0) for n in range(6, 10)]
    session, results = _dims(spark)

    def first_laps(rows):
        df = build_feat_tire_degradation(
            spark.createDataFrame(rows, FACT_LAP_SCHEMA), session, results
        )
        return sorted(tuple(r) for r in df.filter("lap_number <= 5").collect())

    assert first_laps(base) == first_laps(changed)


@pytest.fixture(scope="module")
def synthetic():
    return simulate_laps(n_circuits=4, races_per_circuit=2, n_drivers=6, seed=1)


def test_split_and_perfect_shape_scores_zero(synthetic):
    df, _ = synthetic
    split = evaluation.split_stints(df, n_anchor=3, min_target=5)
    assert len(split.anchor) == 3 * split.anchor.groupby(evaluation.STINT).ngroups
    # A "model" whose shape is the truth minus a per-stint constant forecasts perfectly.
    level = split.anchor.groupby(evaluation.STINT)["lap_time_s"].transform("mean")
    shape_anchor = split.anchor["lap_time_s"].to_numpy() - 7.0
    shape_target = split.target["lap_time_s"].to_numpy() - 7.0
    pred = evaluation.anchored(split, shape_anchor, shape_target)[0]
    assert np.allclose(pred, split.target["lap_time_s"].to_numpy())
    assert level.notna().all()


def test_b0_recovers_compound_ordering(synthetic):
    df, _ = synthetic
    slopes = stint_slopes(df).groupby("compound")["slope"].median()
    # Raw slopes absorb fuel burn (negative), but the ordering survives.
    assert slopes["SOFT"] > slopes["MEDIUM"] > slopes["HARD"]
    model = B0CompoundSlope().fit(df)
    split = evaluation.split_stints(df)
    point, lo, hi = model.predict(split)
    scores = evaluation.score(split.target["lap_time_s"].to_numpy(), point, lo, hi)
    assert scores["coverage_90"] > 0.8


def test_b1_lightgbm_runs(synthetic):
    pytest.importorskip("lightgbm")
    from pitwall.models.baselines import B1LightGBM

    df, _ = synthetic
    train = df[df["race_id"].str.endswith("r0")]
    test = df[df["race_id"].str.endswith("r1")]
    split = evaluation.split_stints(test)
    point, lo, hi = B1LightGBM().fit(train).predict(split)
    assert len(point) == len(split.target) and np.all(lo <= hi + 1e-9)


def _fake_draws(n=200, seed=0):
    rng = np.random.default_rng(seed)
    bc = np.column_stack([rng.normal(m, 0.005, n) for m in (0.09, 0.05, 0.02)])
    beta = bc[:, None, :] + rng.normal(0, 0.01, (n, 2, 3))
    return {
        "fuel_effect": rng.normal(0.035, 0.002, n),
        "beta_compound": bc,
        "circuit_sigma": np.abs(rng.normal(0.02, 0.003, n)),
        "beta": beta,
        "driver_deg_sigma": np.abs(rng.normal(0.005, 0.001, n)),
        "driver_deg": rng.normal(0, 0.005, (n, 2)),
        "gamma_temp": rng.normal(0.005, 0.002, n),
        "delta_traffic": rng.normal(0.4, 0.05, n),
        "sigma": np.full(n, 0.3),
        "nu": np.full(n, 6.0),
        "circuits": ["c0", "c1"],
        "compounds": ["SOFT", "MEDIUM", "HARD"],
        "drivers": ["d0", "d1"],
    }


def test_shape_draws_handle_unseen_circuit_and_driver():
    draws = _fake_draws()
    rows = pd.DataFrame(
        {
            "circuit_id": ["c0", "new", "new"],
            "driver_id": ["d0", "rookie", "rookie"],
            "compound": ["SOFT", "HARD", "HARD"],
            "tyre_life_laps": [10, 10, 20],
            "fuel_proxy": [0, 0, 0],
            "track_temp_c": [30.0, 30.0, 30.0],
            "in_traffic": [False, False, False],
        }
    )
    shape = shape_draws(draws, rows, TempScaler(30.0, 5.0))
    assert shape.shape == (200, 3)
    # Unseen circuit + driver: same deviation on both rows, so lap 20 is exactly 2x lap 10.
    assert np.allclose(shape[:, 2], 2 * shape[:, 1])
    with pytest.raises(ValueError):
        shape_draws(draws, rows.assign(compound="WET"), TempScaler(30.0, 5.0))


def test_predictive_draws_and_summaries(synthetic):
    df, _ = synthetic
    df = df[df["circuit_id"].isin(["c0", "c1"]) & df["driver_id"].isin(["d0", "d1"])]
    split = evaluation.split_stints(df)
    pred = predictive_draws(_fake_draws(), split, TempScaler(35.0, 9.0))
    assert pred.shape == (200, len(split.target))

    table = degradation_table(_fake_draws())
    assert len(table) == 2 * 3 + 3 and (table["lo"] <= table["hi"]).all()
    h = hypothesis_tests(_fake_draws())
    assert h["h1a_p_ordered_overall"] > 0.9 and h["h1c_p_circuit_sd_gt_driver_sd"] > 0.9


@pytest.mark.slow
def test_bayesian_model_recovers_known_parameters():
    """Research plan validity check: fit simulated data, recover the truth."""
    pytest.importorskip("pymc")
    from pitwall.models.synthetic import Truth
    from pitwall.models.tire_degradation import fit, posterior_draws

    df, true_beta = simulate_laps(n_circuits=6, races_per_circuit=2, n_drivers=10, seed=3)
    idata, _ = fit(df, draws=500, tune=500, chains=2, seed=3)
    draws = posterior_draws(idata)
    truth = Truth()

    lo, hi = np.quantile(draws["beta_compound"], [0.03, 0.97], axis=0)
    for j, k in enumerate(draws["compounds"]):
        assert lo[j] - 0.01 <= truth.beta_compound[k] <= hi[j] + 0.01, k
    fuel_lo, fuel_hi = np.quantile(draws["fuel_effect"], [0.03, 0.97])
    assert fuel_lo <= truth.fuel_effect <= fuel_hi

    # Circuit x compound slopes: most true values inside their 90% intervals.
    covered = []
    for i, c in enumerate(draws["circuits"]):
        for j, k in enumerate(draws["compounds"]):
            b_lo, b_hi = np.quantile(draws["beta"][:, i, j], [0.05, 0.95])
            covered.append(b_lo <= true_beta[(c, k)] <= b_hi)
    assert np.mean(covered) >= 0.75
