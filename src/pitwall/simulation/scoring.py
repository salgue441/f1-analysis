"""RQ3 backtest scoring: how good are the simulated finishing-position distributions?

- Ranked probability score (RPS) per driver: squared distance between predicted and
  observed cumulative distributions over positions 1..N, averaged over thresholds. Lower is
  better; the baseline is "finish = grid" (all mass on the grid slot).
- Calibration: P(top-10) reliability by probability bin, and the randomized PIT of the
  observed position (uniform if calibrated).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def position_distribution(positions: np.ndarray, n_positions: int) -> np.ndarray:
    """(n_sims, n_cars) finishing positions -> (n_cars, n_positions) probabilities."""
    n_cars = positions.shape[1]
    probs = np.zeros((n_cars, n_positions))
    for i in range(n_cars):
        counts = np.bincount(positions[:, i] - 1, minlength=n_positions)[:n_positions]
        probs[i] = counts / counts.sum()
    return probs


def rps(probs: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Ranked probability score per row; `observed` positions are 1-based."""
    n_positions = probs.shape[1]
    cdf = np.cumsum(probs, axis=1)[:, :-1]
    obs_cdf = (np.arange(1, n_positions)[None, :] >= observed[:, None]).astype(float)
    return ((cdf - obs_cdf) ** 2).sum(axis=1) / (n_positions - 1)


def degenerate(positions: np.ndarray, n_positions: int) -> np.ndarray:
    probs = np.zeros((len(positions), n_positions))
    probs[np.arange(len(positions)), np.clip(positions, 1, n_positions) - 1] = 1.0
    return probs


def randomized_pit(probs: np.ndarray, observed: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """PIT for a discrete distribution: uniform between F(y-1) and F(y)."""
    cdf = np.cumsum(probs, axis=1)
    upper = cdf[np.arange(len(observed)), observed - 1]
    lower = np.where(observed > 1, cdf[np.arange(len(observed)), observed - 2], 0.0)
    return lower + rng.random(len(observed)) * (upper - lower)


def reliability(p: np.ndarray, hit: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "hit": hit.astype(float)})
    return (
        df.groupby("bin")
        .agg(mean_p=("p", "mean"), observed=("hit", "mean"), n=("hit", "size"))
        .reset_index()
    )


def expected_calibration_error(table: pd.DataFrame) -> float:
    return float(
        (table["n"] * (table["mean_p"] - table["observed"]).abs()).sum() / table["n"].sum()
    )
