"""Typed settings: conf/base.yaml <- conf/<PITWALL_ENV>.yaml <- PITWALL_* env vars."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

CONF_DIR = Path(os.environ.get("PITWALL_CONF_DIR", Path(__file__).resolve().parents[2] / "conf"))


class SeasonRange(BaseModel):
    start: int
    end: int

    def as_list(self) -> list[int]:
        return list(range(self.start, self.end + 1))


class JolpicaSettings(BaseModel):
    base_url: str
    page_size: int = 100
    min_interval_s: float = 8.0
    max_retries: int = 5


class FastF1Settings(BaseModel):
    cache_dir: str
    session_types: list[str] = Field(default_factory=lambda: ["R"])


class SparkSettings(BaseModel):
    master: str = "local[*]"
    conf: dict[str, str] = Field(default_factory=dict)


class QualitySettings(BaseModel):
    max_quarantine_ratio: float = 0.05


class MlflowSettings(BaseModel):
    experiment_prefix: str = "pitwall"


class Settings(BaseModel):
    lakehouse_root: str
    seasons: SeasonRange
    jolpica: JolpicaSettings
    fastf1: FastF1Settings
    spark: SparkSettings = Field(default_factory=SparkSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    mlflow: MlflowSettings = Field(default_factory=MlflowSettings)


# Env var -> dotted settings path. Kept explicit so overrides are discoverable.
ENV_OVERRIDES = {
    "PITWALL_LAKEHOUSE_ROOT": "lakehouse_root",
    "PITWALL_SPARK_MASTER": "spark.master",
    "PITWALL_FASTF1_CACHE_DIR": "fastf1.cache_dir",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = data
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def load_settings(env: str | None = None, conf_dir: Path = CONF_DIR) -> Settings:
    data = yaml.safe_load((conf_dir / "base.yaml").read_text()) or {}
    env = env or os.environ.get("PITWALL_ENV", "local")
    env_file = conf_dir / f"{env}.yaml"
    if env_file.exists():
        data = _deep_merge(data, yaml.safe_load(env_file.read_text()) or {})
    for var, dotted in ENV_OVERRIDES.items():
        if var in os.environ:
            _set_dotted(data, dotted, os.environ[var])
    return Settings.model_validate(data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
