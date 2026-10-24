"""RQ4 models: calibrated pre-race P(podium) and P(points) per driver.

- `GridLogit`: the baseline, a logistic regression on grid position alone.
- `Gbm`: gradient boosting on every pre-race feature, monotone in grid and qualifying.
- `Calibrator`: isotonic or Platt, fitted on the model's *earlier* out-of-sample predictions.
- `PlackettLuce`: a rank model. Each car gets a strength `x @ w`; the finishing order is drawn
  best-first with probability proportional to exp(strength) among the cars left. Its
  probabilities are coherent by construction (P(podium) sums to 3 over a race).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit, logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from pitwall.features.prerace import FEATURES

# Feature -> monotone direction for the boosting model (+1 increasing, -1 decreasing in P).
MONOTONE = {"grid_pos": -1, "quali_position": -1, "quali_gap_pct": -1}
P_CLIP = 1e-4


def _grid_design(df: pd.DataFrame) -> np.ndarray:
    grid = df["grid_pos"].to_numpy(float)
    return np.column_stack([grid, np.log(grid)])


@dataclass
class GridLogit:
    model: LogisticRegression = field(default_factory=lambda: LogisticRegression(C=1e3))

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> GridLogit:
        self.model.fit(_grid_design(df), y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(_grid_design(df))[:, 1]


@dataclass
class Gbm:
    seed: int = 0
    model: HistGradientBoostingClassifier | None = None

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> Gbm:
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            monotonic_cst=[MONOTONE.get(f, 0) for f in FEATURES],
            random_state=self.seed,
        )
        self.model.fit(df[FEATURES].to_numpy(float), y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.predict_proba(df[FEATURES].to_numpy(float))[:, 1]


@dataclass
class Calibrator:
    method: str  # "isotonic" | "platt"
    model: IsotonicRegression | LogisticRegression | None = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator:
        if self.method == "isotonic":
            self.model = IsotonicRegression(y_min=P_CLIP, y_max=1 - P_CLIP, out_of_bounds="clip")
            self.model.fit(p, y)
        else:
            self.model = LogisticRegression(C=1e3).fit(_logit(p)[:, None], y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if isinstance(self.model, IsotonicRegression):
            return self.model.predict(p)
        assert self.model is not None
        return self.model.predict_proba(_logit(p)[:, None])[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, P_CLIP, 1 - P_CLIP))


# --- Plackett-Luce ---------------------------------------------------------------------------


@dataclass
class Standardizer:
    """Column means/SDs from training data; missing values become the training mean (0)."""

    mean: np.ndarray | None = None
    sd: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> Standardizer:
        self.mean = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.sd
        return np.nan_to_num(z, nan=0.0)


def _pl_design(df: pd.DataFrame) -> np.ndarray:
    x = df[FEATURES].to_numpy(float)
    # Grid enters on a log scale too: the gap between P1 and P3 matters more than P15-P17.
    return np.column_stack([x, np.log(df["grid_pos"].to_numpy(float))])


def _race_tensor(z: np.ndarray, race: np.ndarray, finish: np.ndarray):
    """(n_races, max_cars, n_features) design sorted by finishing position, plus a mask."""
    races = pd.unique(race)
    n_max = int(pd.Series(race).value_counts().max())
    xt = np.zeros((len(races), n_max, z.shape[1]))
    mask = np.zeros((len(races), n_max), dtype=bool)
    for i, r in enumerate(races):
        rows = np.flatnonzero(race == r)
        rows = rows[np.argsort(finish[rows], kind="stable")]
        xt[i, : len(rows)] = z[rows]
        mask[i, : len(rows)] = True
    return xt, mask


def pl_negloglik(w: np.ndarray, xt: np.ndarray, mask: np.ndarray, l2: float):
    """Negative PL log-likelihood of the observed orders (+ ridge) and its gradient."""
    # Padded slots (races with fewer cars) are -inf strengths; their NaNs are masked out.
    with np.errstate(invalid="ignore", divide="ignore"):
        return _pl_negloglik(w, xt, mask, l2)


def _pl_negloglik(w: np.ndarray, xt: np.ndarray, mask: np.ndarray, l2: float):
    s = np.where(mask, xt @ w, -np.inf)  # (R, N) strengths in finishing order
    n = s.shape[1]
    upper = np.triu(np.ones((n, n), dtype=bool))  # [j, k]: car k is still running at step j
    # Log-sum-exp over the cars not yet placed at each step j (positions j..N).
    tail = np.where(upper[None], s[:, None, :], -np.inf)  # (R, N, N)
    lse = logsumexp(tail, axis=2)  # (R, N)
    valid = mask
    ll = np.where(valid, s - lse, 0.0).sum()
    weights = np.exp(tail - np.where(valid, lse, 0.0)[:, :, None])  # softmax over the tail
    weights = np.where(valid[:, :, None], weights, 0.0)
    expected = np.einsum("rjk,rkf->rjf", weights, xt)  # E[x] under each step's softmax
    grad = np.where(valid[:, :, None], xt - expected, 0.0).sum(axis=(0, 1))
    return -ll + l2 * w @ w, -grad + 2 * l2 * w


@dataclass
class PlackettLuce:
    l2: float = 1.0
    w: np.ndarray | None = None
    scaler: Standardizer = field(default_factory=Standardizer)

    def fit(self, df: pd.DataFrame) -> PlackettLuce:
        x = _pl_design(df)
        z = self.scaler.fit(x).transform(x)
        xt, mask = _race_tensor(z, df["race_id"].to_numpy(), df["finish_position"].to_numpy())
        res = minimize(
            pl_negloglik,
            np.zeros(z.shape[1]),
            args=(xt, mask, self.l2),
            jac=True,
            method="L-BFGS-B",
        )
        self.w = res.x
        return self

    def strengths(self, df: pd.DataFrame) -> np.ndarray:
        assert self.w is not None
        return self.scaler.transform(_pl_design(df)) @ self.w

    def position_probs(
        self, df: pd.DataFrame, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """(n_cars, n_cars) P(finish in position j) for one race, by Gumbel-max sampling."""
        s = self.strengths(df)
        noisy = s[None, :] + rng.gumbel(size=(n_samples, len(s)))
        pos = np.argsort(np.argsort(-noisy, axis=1), axis=1)  # 0-based finishing position
        probs = np.zeros((len(s), len(s)))
        for j in range(len(s)):
            probs[:, j] = (pos == j).mean(axis=0)
        return probs


# --- Scoring ---------------------------------------------------------------------------------


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
