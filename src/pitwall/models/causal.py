"""RQ2 estimators: does pitting first cause the car behind to end up ahead?

All estimators work on feat_pit_decision (pandas) and report race-clustered uncertainty:
units within a race share safety cars, strategies and weather, so they aren't independent.

- naive difference in means (to show the bias),
- inverse propensity weighting (Hajek), with overlap trimming and cluster bootstrap,
- double ML for the partially linear model with cross-fitting by race (GroupKFold) and a
  cluster-robust influence-function standard error,
- diagnostics: propensity overlap, standardized mean differences before/after weighting,
- refutations: placebo outcome, permuted treatment, E-value for unobserved confounding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

NUMERIC = [
    "gap_s",
    "position",
    "tyre_life",
    "rival_tyre_life",
    "laps_remaining",
    "race_fraction_remaining",
    "track_temp_c",
    "pit_loss_s",
    "rejoin_traffic_cars",
]
CATEGORICAL = ["compound", "rival_compound", "deg_tier"]


@dataclass(frozen=True)
class Estimate:
    method: str
    effect: float
    se: float
    lo: float
    hi: float
    n: int
    n_treated: int

    def as_dict(self) -> dict:
        return asdict(self)


def design_matrix(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    """Numeric covariates (median-imputed) + one-hot categoricals. Pre-decision only.

    `extra` adds numeric columns for robustness checks (e.g. the recent gap trend).
    """
    x = df[NUMERIC + (extra or [])].astype(float)
    x = x.fillna(x.median())
    cats = pd.get_dummies(df[CATEGORICAL].fillna("NA").astype(str), drop_first=True, dtype=float)
    return pd.concat([x, cats], axis=1)


def _cluster_se_mean_diff(y: np.ndarray, t: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """OLS of y on [1, t] with a cluster-robust (CR1) sandwich; returns (effect, se)."""
    x = np.column_stack([np.ones_like(t, dtype=float), t.astype(float)])
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    meat = np.zeros((2, 2))
    codes, _ = pd.factorize(groups)
    for g in range(codes.max() + 1):
        s = x[codes == g].T @ resid[codes == g]
        meat += np.outer(s, s)
    n, k, g_count = len(y), 2, codes.max() + 1
    adj = g_count / (g_count - 1) * (n - 1) / (n - k)
    cov = adj * xtx_inv @ meat @ xtx_inv
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def naive(df: pd.DataFrame, outcome: str, cluster: str = "race_id") -> Estimate:
    y, t = df[outcome].to_numpy(float), df["treated"].to_numpy(int)
    effect, se = _cluster_se_mean_diff(y, t, df[cluster].to_numpy())
    return Estimate(
        "naive", effect, se, effect - 1.96 * se, effect + 1.96 * se, len(df), int(t.sum())
    )


def _classifier(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=40, random_state=seed
    )


def _regressor(seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=40, random_state=seed
    )


def cross_fit(
    df: pd.DataFrame,
    outcome: str,
    folds: int = 5,
    seed: int = 0,
    cluster: str = "race_id",
    extra: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold nuisances: propensity e(x) = P(T=1|X) and outcome g(x) = E[Y|X].

    Folds split by race, so a race's own units never train the models that predict them.
    """
    from sklearn.model_selection import GroupKFold

    x = design_matrix(df, extra).to_numpy()
    t, y = df["treated"].to_numpy(int), df[outcome].to_numpy(float)
    groups = df[cluster].to_numpy()
    e_hat, g_hat = np.empty(len(df)), np.empty(len(df))
    for train, test in GroupKFold(n_splits=folds).split(x, t, groups):
        e_hat[test] = _classifier(seed).fit(x[train], t[train]).predict_proba(x[test])[:, 1]
        g_hat[test] = _regressor(seed).fit(x[train], y[train]).predict(x[test])
    return e_hat, g_hat


def overlap_mask(e_hat: np.ndarray, trim: float = 0.05) -> np.ndarray:
    return (e_hat >= trim) & (e_hat <= 1 - trim)


def _hajek(y: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
    w1, w0 = t / e, (1 - t) / (1 - e)
    return float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())


def ipw(
    df: pd.DataFrame,
    outcome: str,
    e_hat: np.ndarray,
    trim: float = 0.05,
    n_boot: int = 500,
    seed: int = 0,
    cluster: str = "race_id",
) -> Estimate:
    """Hajek IPW ATE on the overlap region; CI from a race-cluster bootstrap (nuisances fixed)."""
    keep = overlap_mask(e_hat, trim)
    sub, e = df[keep], e_hat[keep]
    y, t = sub[outcome].to_numpy(float), sub["treated"].to_numpy(int)
    effect = _hajek(y, t, e)
    rng = np.random.default_rng(seed)
    codes, uniques = pd.factorize(sub[cluster])
    index_by_cluster = [np.flatnonzero(codes == g) for g in range(len(uniques))]
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate(
            [index_by_cluster[g] for g in rng.integers(0, len(uniques), len(uniques))]
        )
        if t[pick].sum() == 0 or t[pick].sum() == len(pick):
            continue
        boots.append(_hajek(y[pick], t[pick], e[pick]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return Estimate(
        "ipw", effect, float(np.std(boots)), float(lo), float(hi), len(sub), int(t.sum())
    )


def dml(
    df: pd.DataFrame,
    outcome: str,
    e_hat: np.ndarray,
    g_hat: np.ndarray,
    trim: float = 0.05,
    cluster: str = "race_id",
) -> Estimate:
    """Partially linear DML (partialling out) with a cluster-robust influence-function SE."""
    keep = overlap_mask(e_hat, trim)
    y = df[outcome].to_numpy(float)[keep]
    t = df["treated"].to_numpy(float)[keep]
    v = t - e_hat[keep]  # treatment residual
    u = y - g_hat[keep]  # outcome residual (g is E[Y|X], not E[Y|X,T]: partialling out)
    theta = float((v * u).sum() / (v * v).sum())
    psi = (u - theta * v) * v / np.mean(v * v)
    codes, _ = pd.factorize(df[cluster].to_numpy()[keep])
    cluster_sums = np.bincount(codes, psi)
    n, g_count = len(psi), len(cluster_sums)
    se = float(np.sqrt((cluster_sums**2).sum() * g_count / (g_count - 1)) / n)
    return Estimate("dml", theta, se, theta - 1.96 * se, theta + 1.96 * se, n, int(t.sum()))


def smd(df: pd.DataFrame, weights: np.ndarray | None = None) -> pd.Series:
    """Standardized mean difference treated vs control per covariate (|SMD| < 0.1 = balanced)."""
    x = design_matrix(df)
    t = df["treated"].to_numpy(bool)
    w = np.ones(len(df)) if weights is None else weights

    def wmean(a: np.ndarray, ww: np.ndarray) -> float:
        return float((a * ww).sum() / ww.sum())

    def wvar(a: np.ndarray, ww: np.ndarray) -> float:
        m = wmean(a, ww)
        return float((ww * (a - m) ** 2).sum() / ww.sum())

    out = {}
    for col in x.columns:
        a = x[col].to_numpy(float)
        m1, m0 = wmean(a[t], w[t]), wmean(a[~t], w[~t])
        pooled = np.sqrt((wvar(a[t], np.ones(t.sum())) + wvar(a[~t], np.ones((~t).sum()))) / 2)
        out[col] = (m1 - m0) / pooled if pooled > 0 else 0.0
    return pd.Series(out)


def ipw_weights(t: np.ndarray, e_hat: np.ndarray) -> np.ndarray:
    return np.where(t == 1, 1 / e_hat, 1 / (1 - e_hat))


def e_value(rr: float) -> float:
    """VanderWeele & Ding E-value for a risk ratio (applied to whichever side of 1 it is)."""
    rr = rr if rr >= 1 else 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


def risk_ratio(df: pd.DataFrame, outcome: str, e_hat: np.ndarray, trim: float = 0.05) -> float:
    keep = overlap_mask(e_hat, trim)
    y, t, e = df[outcome].to_numpy(float)[keep], df["treated"].to_numpy(int)[keep], e_hat[keep]
    w1, w0 = t / e, (1 - t) / (1 - e)
    p1, p0 = (w1 * y).sum() / w1.sum(), (w0 * y).sum() / w0.sum()
    return float(p1 / p0) if p0 > 0 else float("inf")


def permuted_treatment_refuter(
    df: pd.DataFrame, outcome: str, n_perm: int = 20, seed: int = 0, cluster: str = "race_id"
) -> dict[str, float]:
    """Shuffle treatment within race; DML on fake treatment should give ~0."""
    rng = np.random.default_rng(seed)
    effects = []
    for i in range(n_perm):
        fake = df.copy()
        fake["treated"] = fake.groupby(cluster)["treated"].transform(
            lambda s: rng.permutation(s.to_numpy())
        )
        e_hat, g_hat = cross_fit(fake, outcome, seed=seed + i)
        effects.append(dml(fake, outcome, e_hat, g_hat).effect)
    return {
        "mean_effect": float(np.mean(effects)),
        "sd_effect": float(np.std(effects)),
        "n_perm": n_perm,
    }


def econml_cross_check(df: pd.DataFrame, outcome: str, seed: int = 0) -> dict[str, float] | None:
    """LinearDML from EconML with race-grouped cross-fitting, if EconML is installed."""
    try:
        from econml.dml import LinearDML
    except ImportError:
        return None
    x = design_matrix(df).to_numpy()
    est = LinearDML(
        model_y=_regressor(seed),
        model_t=_classifier(seed),
        discrete_treatment=True,
        cv=5,
        random_state=seed,
    )
    est.fit(
        df[outcome].to_numpy(float),
        df["treated"].to_numpy(int),
        X=None,
        W=x,
        groups=df["race_id"].to_numpy(),
    )
    lo, hi = est.ate_interval(alpha=0.05)
    return {"effect": float(est.ate()), "lo": float(lo), "hi": float(hi)}
