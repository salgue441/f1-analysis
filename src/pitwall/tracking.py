"""MLflow tracking with the reproducibility metadata SAD §9.4 requires.

Every run logs the git SHA (and whether the tree was dirty), the Delta version of each table
it read, the random seed and a snapshot of the configuration.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pitwall.config import get_settings


def _mlflow_dir() -> Path:
    return Path(get_settings().lakehouse_root).parent / "mlflow"


def tracking_uri() -> str:
    """The compose MLflow server when set, else the same SQLite store it serves from."""
    default = f"sqlite:///{_mlflow_dir() / 'mlflow.db'}"
    return os.environ.get("MLFLOW_TRACKING_URI", default)


def git_state() -> dict[str, str]:
    def git(*args: str) -> str:
        out = subprocess.run(
            ["git", "-c", "safe.directory=*", *args], capture_output=True, text=True, check=False
        )
        return out.stdout.strip()

    return {
        "git_sha": git("rev-parse", "HEAD") or "unknown",
        "git_dirty": str(bool(git("status", "--porcelain"))),
    }


def delta_versions(paths: dict[str, str]) -> dict[str, int]:
    from delta.tables import DeltaTable

    from pitwall.spark import get_spark

    spark = get_spark()
    return {
        name: int(DeltaTable.forPath(spark, path).history(1).select("version").first()[0])
        for name, path in paths.items()
    }


@contextmanager
def tracked_run(
    experiment: str, params: dict[str, Any], tables: dict[str, str], seed: int
) -> Iterator[Any]:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    name = f"{get_settings().mlflow.experiment_prefix}-{experiment}"
    if mlflow.get_experiment_by_name(name) is None:
        artifacts = _mlflow_dir() / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(name)
    with mlflow.start_run() as run:
        mlflow.set_tags({**git_state(), "seed": str(seed)})
        mlflow.log_params({**params, "seed": seed})
        mlflow.log_params({f"delta_version.{k}": v for k, v in delta_versions(tables).items()})
        mlflow.log_dict(json.loads(get_settings().model_dump_json()), "config.json")
        yield run
