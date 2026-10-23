"""Distribute RQ3 simulations over Spark (ADR-0003, SAD §7.3).

Race set-ups (field, strategies, parameter draws) are broadcast once. The task table holds
small `(race, [driver,] batch, seed)` rows, so work splits into independent batches that
executors run with the NumPy simulator via `mapInPandas`. Tasks return sums (position
counts, points totals), never individual simulations, so 10^6+ races aggregate in Spark.

Seeds depend only on (race, batch): every strategy evaluated for a race in a batch sees the
same random stream (common random numbers), so strategy comparisons are paired.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pitwall.simulation.field import Car, FieldParams, points_for, simulate_field
from pitwall.simulation.race_simulator import Strategy

COUNTS_SCHEMA = "race_id string, car_index int, position int, count long"
DECISION_SCHEMA = (
    "race_id string, car_index int, split int, strategy string, n long, points_sum double, "
    "points_sq_sum double, gain_sum double, gain_sq_sum double, n_better long, n_worse long, "
    "position_sum double"
)


@dataclass(frozen=True)
class RaceSetup:
    race_id: str
    cars: tuple[Car, ...]
    params: FieldParams


def _batches(n_sims: int, batch_size: int) -> list[tuple[int, int]]:
    out, batch = [], 0
    while n_sims > 0:
        out.append((batch, min(batch_size, n_sims)))
        n_sims -= batch_size
        batch += 1
    return out


def _seed(seed: int, race_index: int, batch: int) -> int:
    return seed + 1_000_003 * race_index + batch


def simulate_on_spark(
    spark: SparkSession,
    races: list[RaceSetup],
    n_sims: int,
    batch_size: int = 2000,
    seed: int = 0,
    partitions: int | None = None,
) -> pd.DataFrame:
    """Finishing-position counts per race and car: race_id, car_index, position, count."""
    shared = spark.sparkContext.broadcast({r.race_id: r for r in races})
    rows = [
        (r.race_id, b, _seed(seed, i, b), n)
        for i, r in enumerate(races)
        for b, n in _batches(n_sims, batch_size)
    ]
    tasks = spark.createDataFrame(rows, "race_id string, batch int, seed long, n int")

    def run(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        lookup = shared.value
        for pdf in batches:
            for task in pdf.itertuples(index=False):
                race = lookup[task.race_id]
                n_cars = len(race.cars)
                pos = simulate_field(
                    list(race.cars), race.params, int(task.n), np.random.default_rng(int(task.seed))
                )
                flat = np.arange(n_cars)[None, :] * n_cars + (pos - 1)
                counts = np.bincount(flat.ravel(), minlength=n_cars * n_cars)
                nz = np.flatnonzero(counts)
                yield pd.DataFrame(
                    {
                        "race_id": task.race_id,
                        "car_index": (nz // n_cars).astype("int32"),
                        "position": (nz % n_cars + 1).astype("int32"),
                        "count": counts[nz].astype("int64"),
                    }
                )

    result = (
        tasks.repartition(partitions or spark.sparkContext.defaultParallelism)
        .mapInPandas(run, COUNTS_SCHEMA)
        .groupBy("race_id", "car_index", "position")
        .agg(F.sum("count").alias("count"))
        .toPandas()
    )
    shared.unpersist()
    return result


def position_probs(counts: pd.DataFrame, race_id: str, n_cars: int) -> np.ndarray:
    """(n_cars, n_positions) probability matrix for one race."""
    part = counts[counts["race_id"] == race_id]
    probs = np.zeros((n_cars, n_cars))
    probs[part["car_index"], part["position"] - 1] = part["count"]
    return probs / probs.sum(axis=1, keepdims=True)


ACTUAL_PREFIX = "actual:"


def _target_run(
    race: RaceSetup, car: int, strategy: Strategy, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Points and finishing position of one car under `strategy`, everyone else as set."""
    cars = list(race.cars)
    cars[car] = replace(cars[car], strategy=strategy)
    pos = simulate_field(cars, race.params, n, np.random.default_rng(seed))[:, car]
    return points_for(pos), pos


def evaluate_strategies_on_spark(
    spark: SparkSession,
    races: list[RaceSetup],
    targets: dict[str, list[int]],
    candidates: dict[str, list[Strategy]],
    n_sims: int,
    batch_size: int = 500,
    seed: int = 0,
    partitions: int | None = None,
) -> pd.DataFrame:
    """For each race and target car: every candidate strategy vs the car's actual one.

    Returns raw sums per (race, car, split, strategy); `split` is the batch parity, so a
    best strategy can be chosen on split 0 and scored on split 1 without the optimism of
    picking the max of noisy estimates. Use `summarize` for means, SEs and probabilities.
    """
    shared = spark.sparkContext.broadcast(
        {"races": {r.race_id: r for r in races}, "candidates": candidates}
    )
    rows = [
        (r.race_id, car, b, _seed(seed, i, b), n)
        for i, r in enumerate(races)
        for car in targets.get(r.race_id, [])
        for b, n in _batches(n_sims, batch_size)
    ]
    tasks = spark.createDataFrame(
        rows, "race_id string, car_index int, batch int, seed long, n int"
    )

    def run(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        value = shared.value
        for pdf in batches:
            for task in pdf.itertuples(index=False):
                race = value["races"][task.race_id]
                car, n, seed = int(task.car_index), int(task.n), int(task.seed)
                actual = race.cars[car].strategy
                base, _ = _target_run(race, car, actual, n, seed)
                out = []
                # The actual strategy is the paired baseline (gain 0), labelled apart from
                # any identical candidate.
                labelled = [(f"{ACTUAL_PREFIX}{actual.name}", actual)] + [
                    (s.name, s) for s in value["candidates"][task.race_id]
                ]
                for label, strategy in labelled:
                    pts, pos = _target_run(race, car, strategy, n, seed)
                    gain = pts - base
                    out.append(
                        (
                            task.race_id,
                            car,
                            int(task.batch) % 2,
                            label,
                            len(pts),
                            pts.sum(),
                            (pts**2).sum(),
                            gain.sum(),
                            (gain**2).sum(),
                            int((gain > 0).sum()),
                            int((gain < 0).sum()),
                            pos.sum(),
                        )
                    )
                yield pd.DataFrame(out, columns=[c.split()[0] for c in DECISION_SCHEMA.split(", ")])

    sums = (
        tasks.repartition(partitions or spark.sparkContext.defaultParallelism)
        .mapInPandas(run, DECISION_SCHEMA)
        .groupBy("race_id", "car_index", "split", "strategy")
        .agg(*[F.sum(c).alias(c) for c in SUM_COLUMNS])
        .toPandas()
    )
    shared.unpersist()
    return sums


SUM_COLUMNS = [
    "n",
    "points_sum",
    "points_sq_sum",
    "gain_sum",
    "gain_sq_sum",
    "n_better",
    "n_worse",
    "position_sum",
]


def summarize(sums: pd.DataFrame, split: int | None = None) -> pd.DataFrame:
    """Means and paired-gain SEs per (race, car, strategy), from one split or both pooled."""
    if split is not None:
        sums = sums[sums["split"] == split]
    sums = sums.groupby(["race_id", "car_index", "strategy"], as_index=False)[SUM_COLUMNS].sum()
    n = sums["n"]
    mean_gain = sums["gain_sum"] / n
    var_gain = (sums["gain_sq_sum"] / n - mean_gain**2) * n / (n - 1)
    return pd.DataFrame(
        {
            "race_id": sums["race_id"],
            "car_index": sums["car_index"],
            "strategy": sums["strategy"],
            "is_actual": sums["strategy"].str.startswith(ACTUAL_PREFIX),
            "n_sims": n,
            "expected_points": sums["points_sum"] / n,
            "expected_position": sums["position_sum"] / n,
            "points_gain_vs_actual": mean_gain,
            "points_gain_se": np.sqrt(var_gain.clip(lower=0) / n),
            "p_better_than_actual": sums["n_better"] / n,
            "p_worse_than_actual": sums["n_worse"] / n,
        }
    )


def scaling_benchmark(
    spark: SparkSession,
    race: RaceSetup,
    n_sims: int,
    parallelism: list[int],
    batch_size: int = 2000,
) -> pd.DataFrame:
    """Wall time for `n_sims` races at different task parallelism (partitions = cores used).

    In local mode the partition count bounds how many cores work at once, so this is the
    single-node stand-in for adding executors; on a cluster the same code scales out.
    """
    rows = []
    for p in parallelism:
        start = time.perf_counter()
        counts = simulate_on_spark(spark, [race], n_sims, batch_size, seed=1, partitions=p)
        elapsed = time.perf_counter() - start
        simulated = int(counts["count"].sum() // len(race.cars))
        rows.append(
            {
                "parallelism": p,
                "n_sims": simulated,
                "seconds": elapsed,
                "sims_per_second": simulated / elapsed,
            }
        )
    df = pd.DataFrame(rows)
    df["speedup"] = df["seconds"].iloc[0] / df["seconds"]
    return df
