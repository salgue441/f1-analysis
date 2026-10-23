"""Lightweight row-level expectations with quarantine (SAD §9.1).

Each expectation is a SQL boolean expression that must hold for a row. NULL results
count as failures. `error` failures send the row to quarantine; `warn` failures are
only counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F


@dataclass(frozen=True)
class Expectation:
    name: str
    condition: str
    severity: Literal["error", "warn"] = "error"


@dataclass
class QualityReport:
    total_rows: int
    quarantined_rows: int
    failures: dict[str, int] = field(default_factory=dict)

    @property
    def quarantine_ratio(self) -> float:
        return self.quarantined_rows / self.total_rows if self.total_rows else 0.0


class QualityGateError(RuntimeError):
    pass


def _flag(name: str) -> str:
    return f"_dq_{name}"


def apply_expectations(
    df: DataFrame,
    expectations: list[Expectation],
    max_quarantine_ratio: float | None = None,
) -> tuple[DataFrame, DataFrame, QualityReport]:
    """Split `df` into (valid, quarantined) rows and report failure counts per expectation."""
    flagged = df
    for exp in expectations:
        flagged = flagged.withColumn(
            _flag(exp.name), F.coalesce(F.expr(exp.condition), F.lit(False))
        )

    failed_names = (
        F.filter(
            F.array(*[F.when(~F.col(_flag(e.name)), F.lit(e.name)) for e in expectations]),
            lambda x: x.isNotNull(),
        )
        if expectations
        else F.array().cast("array<string>")
    )

    error_flags: list[Column] = [
        ~F.col(_flag(e.name)) for e in expectations if e.severity == "error"
    ]
    rejected = F.lit(False)
    for flag in error_flags:
        rejected = rejected | flag

    flagged = flagged.withColumn("_failed_expectations", failed_names).withColumn(
        "_is_rejected", rejected
    )

    stats = flagged.agg(
        F.count(F.lit(1)).alias("_total"),
        F.sum(F.col("_is_rejected").cast("int")).alias("_quarantined"),
        *[F.sum((~F.col(_flag(e.name))).cast("int")).alias(e.name) for e in expectations],
    ).first()
    report = QualityReport(
        total_rows=stats["_total"] or 0,
        quarantined_rows=stats["_quarantined"] or 0,
        failures={e.name: stats[e.name] or 0 for e in expectations},
    )

    dq_cols = [_flag(e.name) for e in expectations]
    valid = flagged.filter(~F.col("_is_rejected")).drop(
        *dq_cols, "_failed_expectations", "_is_rejected"
    )
    quarantined = (
        flagged.filter(F.col("_is_rejected"))
        .drop(*dq_cols, "_is_rejected")
        .withColumn("_quarantined_at", F.current_timestamp())
    )

    if max_quarantine_ratio is not None and report.quarantine_ratio > max_quarantine_ratio:
        raise QualityGateError(
            f"{report.quarantined_rows}/{report.total_rows} rows quarantined "
            f"({report.quarantine_ratio:.1%} > {max_quarantine_ratio:.1%}): {report.failures}"
        )
    return valid, quarantined, report
