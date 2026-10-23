"""SparkSession factory with Delta Lake, AQE and UTC configured."""

from __future__ import annotations

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from pitwall.config import Settings, get_settings


def get_spark(app_name: str = "pitwall", settings: Settings | None = None) -> SparkSession:
    settings = settings or get_settings()
    builder = (
        SparkSession.builder.appName(app_name)
        .master(settings.spark.master)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    for key, value in settings.spark.conf.items():
        builder = builder.config(key, value)
    return configure_spark_with_delta_pip(builder).getOrCreate()
