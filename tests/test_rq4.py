import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from pitwall.features.prerace import FEATURES, build_feat_prerace
from pitwall.models.prerace import (
    Calibrator,
    PlackettLuce,
    _race_tensor,
    pl_negloglik,
)

RESULT_COLS = [
    "season",
    "round",
    "session_type",
    "driver_id",
    "constructor_id",
    "grid",
    "finish_position",
    "points",
    "is_classified",
    "best_lap_time_s",
]


def _race(season, rnd, rows):
    """rows: (driver, constructor, grid, finish, points, classified)."""
    return [(season, rnd, "R", d, c, float(g), f, float(p), ok, None) for d, c, g, f, p, ok in rows]


def _quali(season, rnd, times):
    return [
        (season, rnd, "Q", d, c, None, pos, 0.0, True, t)
        for pos, (d, c, t) in enumerate(times, start=1)
    ]


@pytest.fixture
def inputs(spark):
    results = (
        _race(2023, 1, [("ver", "rb", 1, 1, 25, True), ("ham", "merc", 2, 2, 18, True)])
        + _race(2023, 2, [("ver", "rb", 2, 2, 18, True), ("ham", "merc", 1, 1, 25, True)])
        + _race(2024, 1, [("ver", "rb", 0, 1, 25, True), ("ham", "merc", 1, 2, 18, False)])
        + _quali(2024, 1, [("ham", "merc", 90.0), ("ver", "rb", 90.9)])
    )
    sessions = [
        (2023, 1, "R", "bahrain", "ground_effect"),
        (2023, 2, "R", "jeddah", "ground_effect"),
        (2024, 1, "R", "bahrain", "ground_effect"),
    ]
    return (
        spark.createDataFrame(results, RESULT_COLS),
        spark.createDataFrame(sessions, ["season", "round", "session_type", "circuit_id", "era"]),
        spark.createDataFrame(
            [("rb", "rb"), ("merc", "merc")], ["constructor_id", "lineage_group"]
        ),
    )


def _features(inputs) -> pd.DataFrame:
    return build_feat_prerace(*inputs).toPandas().set_index(["race_id", "driver_id"])


def test_form_uses_only_earlier_races(inputs):
    feat = _features(inputs)
    ver = feat.loc[("2024_01", "ver")]
    assert ver["driver_starts"] == 2
    assert ver["driver_points_l5"] == pytest.approx((25 + 18) / 2)
    assert ver["driver_finish_l5"] == pytest.approx(1.5)
    assert ver["team_points_l5"] == pytest.approx((25 + 18) / 2)
    first = feat.loc[("2023_01", "ver")]
    assert first["driver_starts"] == 0
    assert np.isnan(first["driver_points_l5"])


def test_features_ignore_the_race_result(inputs, spark):
    results, sessions, constructors = inputs
    before = _features(inputs).loc["2024_01", FEATURES]
    changed = (
        results.replace(25.0, 0.0, subset=["points"])
        .filter("season = 2024")
        .unionByName(results.filter("season < 2024"))
    )
    after = _features((changed, sessions, constructors)).loc["2024_01", FEATURES]
    pd.testing.assert_frame_equal(before, after)


def test_pit_lane_qualifying_and_circuit_history(inputs):
    feat = _features(inputs)
    ver, ham = feat.loc[("2024_01", "ver")], feat.loc[("2024_01", "ham")]
    assert ver["from_pit_lane"] == 1 and ver["grid_pos"] == 3  # two cars: behind slot 2
    assert ham["quali_gap_pct"] == pytest.approx(0.0)
    assert ver["quali_gap_pct"] == pytest.approx(1.0)
    # Bahrain 2023: both cars finished where they started -> 0; 2023 itself has no history.
    assert ver["circuit_pos_change"] == pytest.approx(0.0)
    assert np.isnan(feat.loc[("2023_01", "ver")]["circuit_pos_change"])
    assert bool(ham["podium"]) and bool(ham["scored_points"])


def _pl_data(n_races=150, n_cars=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        x = rng.normal(size=(n_cars, 2))
        strength = 1.5 * x[:, 0] - 0.5 * x[:, 1]
        order = np.argsort(-(strength + rng.gumbel(size=n_cars)))
        finish = np.empty(n_cars, int)
        finish[order] = np.arange(1, n_cars + 1)
        rows += [(f"r{r}", finish[i], *x[i]) for i in range(n_cars)]
    return pd.DataFrame(rows, columns=["race_id", "finish_position", "a", "b"])


def test_pl_gradient_matches_finite_differences():
    df = _pl_data(n_races=5, n_cars=6)
    z = df[["a", "b"]].to_numpy()
    xt, mask = _race_tensor(z, df["race_id"].to_numpy(), df["finish_position"].to_numpy())
    w = np.array([0.3, -0.2])
    _, grad = pl_negloglik(w, xt, mask, 0.1)
    numeric = approx_fprime(w, lambda v: pl_negloglik(v, xt, mask, 0.1)[0], 1e-6)
    np.testing.assert_allclose(grad, numeric, rtol=1e-4)


def test_pl_recovers_strengths_and_gives_coherent_probabilities(monkeypatch):
    import pitwall.models.prerace as m

    df = _pl_data()
    monkeypatch.setattr(m, "_pl_design", lambda d: d[["a", "b"]].to_numpy(float))
    model = PlackettLuce(l2=0.01).fit(df)
    w = model.w / model.scaler.sd  # back to the raw feature scale
    np.testing.assert_allclose(w, [1.5, -0.5], atol=0.2)
    probs = model.position_probs(df[df["race_id"] == "r0"], 4000, np.random.default_rng(1))
    np.testing.assert_allclose(probs.sum(axis=0), 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibration_fixes_overconfidence(method):
    rng = np.random.default_rng(0)
    true = rng.uniform(0.05, 0.6, 5000)
    y = rng.random(5000) < true
    overconfident = np.clip(true * 1.6, 0, 1)
    cal = Calibrator(method).fit(overconfident[:4000], y[:4000])
    fixed = cal.predict(overconfident[4000:])
    assert np.mean(np.abs(fixed - true[4000:])) < np.mean(
        np.abs(overconfident[4000:] - true[4000:])
    )
