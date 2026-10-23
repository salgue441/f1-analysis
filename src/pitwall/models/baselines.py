"""RQ1 baselines, scored with the same anchored protocol as the Bayesian model.

- B0: per-stint OLS slope of lap time on tyre age, averaged (median) by compound. The shape
  is slope x tyre age; fuel burn is folded into the slope, as a naive analyst would.
- B1: LightGBM on the same features, predicting each lap's offset from its stint anchor.
  Its interval comes from two quantile models (5%, 95%).

Both get intervals from the training data only, so 90% coverage is a fair comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall.models.evaluation import STINT, StintSplit, anchored, split_stints


def stint_slopes(df: pd.DataFrame, min_laps: int = 5) -> pd.DataFrame:
    """OLS slope of lap time on tyre life within each stint (clean laps only)."""

    def slope(g: pd.DataFrame) -> float:
        t, y = g["tyre_life_laps"].to_numpy(float), g["lap_time_s"].to_numpy(float)
        return float(np.polyfit(t, y, 1)[0]) if len(g) >= min_laps and t.std() > 0 else np.nan

    out = df.groupby([*STINT, "compound"])[["tyre_life_laps", "lap_time_s"]].apply(slope)
    return out.rename("slope").reset_index().dropna()


class B0CompoundSlope:
    def fit(self, train: pd.DataFrame, n_anchor: int = 3) -> B0CompoundSlope:
        self.slopes = stint_slopes(train).groupby("compound")["slope"].median().to_dict()
        # Interval from the training residuals of the same anchored forecast.
        split = split_stints(train, n_anchor)
        resid = split.target["lap_time_s"].to_numpy() - self._point(split)
        self.resid_q = np.quantile(resid, [0.05, 0.95])
        return self

    def _shape(self, df: pd.DataFrame) -> np.ndarray:
        slope = df["compound"].map(self.slopes).fillna(np.nanmedian(list(self.slopes.values())))
        return (slope * df["tyre_life_laps"]).to_numpy(float)

    def _point(self, split: StintSplit) -> np.ndarray:
        return anchored(split, self._shape(split.anchor), self._shape(split.target))[0]

    def predict(self, split: StintSplit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        point = self._point(split)
        return point, point + self.resid_q[0], point + self.resid_q[1]


FEATURES = [
    "d_tyre",
    "d_fuel",
    "tyre_life_laps",
    "track_temp_c",
    "in_traffic",
    "compound",
    "circuit_id",
]


def _relative(split: StintSplit) -> pd.DataFrame:
    """Target-lap features relative to the stint anchor, and the target offset."""
    level = split.anchor.groupby(STINT).agg(
        y0=("lap_time_s", "mean"), t0=("tyre_life_laps", "mean"), f0=("fuel_proxy", "mean")
    )
    rel = split.target.join(level, on=STINT)
    rel["d_tyre"] = rel["tyre_life_laps"] - rel["t0"]
    rel["d_fuel"] = rel["fuel_proxy"] - rel["f0"]
    rel["dy"] = rel["lap_time_s"] - rel["y0"]
    rel["in_traffic"] = rel["in_traffic"].astype(int)
    for col in ["compound", "circuit_id"]:
        rel[col] = rel[col].astype("category")
    return rel


class B1LightGBM:
    def __init__(self, seed: int = 42, n_anchor: int = 3):
        self.seed, self.n_anchor = seed, n_anchor

    def fit(self, train: pd.DataFrame) -> B1LightGBM:
        import lightgbm as lgb

        rel = _relative(split_stints(train, self.n_anchor))
        self.categories = {c: rel[c].cat.categories for c in ["compound", "circuit_id"]}
        params = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "random_state": self.seed,
            "verbose": -1,
        }
        x = rel[FEATURES]
        self.models = {
            "mid": lgb.LGBMRegressor(objective="l2", **params).fit(x, rel["dy"]),
            "lo": lgb.LGBMRegressor(objective="quantile", alpha=0.05, **params).fit(x, rel["dy"]),
            "hi": lgb.LGBMRegressor(objective="quantile", alpha=0.95, **params).fit(x, rel["dy"]),
        }
        return self

    def predict(self, split: StintSplit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rel = _relative(split)
        for col, cats in self.categories.items():
            rel[col] = pd.Categorical(rel[col], categories=cats)
        x = rel[FEATURES]
        y0 = rel["y0"].to_numpy()
        return tuple(y0 + self.models[k].predict(x) for k in ("mid", "lo", "hi"))  # type: ignore[return-value]
