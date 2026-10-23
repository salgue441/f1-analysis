"""RQ1 evaluation protocol (research plan §RQ1, fixed before modeling).

Held-out stints are forecast from their first `n_anchor` laps: the anchor fixes the level
(race pace, driver, car), the model supplies the shape (tyre wear, fuel burn, temperature,
traffic). Every model is scored the same way:

    forecast(lap) = mean(y[anchor]) + shape(lap) - mean(shape[anchor])

Metrics: RMSE, MAE, and coverage/width of the 90% predictive interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STINT = ["race_id", "driver_id", "stint_number"]


@dataclass(frozen=True)
class StintSplit:
    anchor: pd.DataFrame  # first n_anchor laps of each stint
    target: pd.DataFrame  # the remaining laps to forecast


def split_stints(df: pd.DataFrame, n_anchor: int = 3, min_target: int = 5) -> StintSplit:
    """Order each stint by lap and split it; stints too short to score are dropped."""
    ordered = df.sort_values([*STINT, "lap_number"]).copy()
    ordered["_pos"] = ordered.groupby(STINT).cumcount()
    size = ordered.groupby(STINT)["lap_number"].transform("size")
    ordered = ordered[size >= n_anchor + min_target]
    return StintSplit(
        anchor=ordered[ordered["_pos"] < n_anchor].drop(columns="_pos"),
        target=ordered[ordered["_pos"] >= n_anchor].drop(columns="_pos"),
    )


def anchored(split: StintSplit, shape_anchor: np.ndarray, shape_target: np.ndarray) -> np.ndarray:
    """Apply the anchoring formula to shape draws: arrays (n_draws, n_rows) or (n_rows,).

    Returns forecasts with the same leading draw axis for the target rows.
    """
    sa = np.atleast_2d(shape_anchor)
    st = np.atleast_2d(shape_target)
    anchor_key = _stint_codes(split.anchor, split)
    target_key = _stint_codes(split.target, split)
    n_stints = int(max(anchor_key.max(), target_key.max())) + 1

    y_level = np.bincount(anchor_key, split.anchor["lap_time_s"].to_numpy(float), n_stints)
    count = np.bincount(anchor_key, minlength=n_stints)
    y_level = y_level / count
    shape_level = np.stack([np.bincount(anchor_key, row, n_stints) for row in sa]) / count
    return y_level[target_key][None, :] + st - shape_level[:, target_key]


def _stint_codes(part: pd.DataFrame, split: StintSplit) -> np.ndarray:
    all_ids = pd.concat([split.anchor[STINT], split.target[STINT]]).drop_duplicates()
    index = pd.MultiIndex.from_frame(all_ids)
    return index.get_indexer(pd.MultiIndex.from_frame(part[STINT]))


def score(y: np.ndarray, point: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    err = point - y
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "coverage_90": float(np.mean((y >= lo) & (y <= hi))),
        "interval_width_90": float(np.mean(hi - lo)),
        "n_laps": int(len(y)),
    }


def score_draws(y: np.ndarray, draws: np.ndarray) -> dict[str, float]:
    """Score predictive draws (n_draws, n_rows): median point forecast, 5-95% interval."""
    lo, mid, hi = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
    return score(y, mid, lo, hi)
