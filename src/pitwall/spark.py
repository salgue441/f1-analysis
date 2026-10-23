"""SparkSession factory with Delta Lake, AQE and UTC configured.

The same factory serves local Spark and managed platforms; only `conf/<env>.yaml` changes
(SAD §8.2): an empty `spark.master` leaves the platform's master in place, and
`delta_pip: false` uses the Delta jars the platform ships instead of downloading them.
"""

from __future__ import annotations

from pyspark.sql import SparkSession

from pitwall.config import Settings, get_settings


def get_spark(app_name: str = "pitwall", settings: Settings | None = None) -> SparkSession:
    settings = settings or get_settings()
    builder = SparkSession.builder.appName(app_name)
    if settings.spark.master:
        builder = builder.master(settings.spark.master)
    builder = (
        builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    for key, value in settings.spark.conf.items():
        builder = builder.config(key, value)
    if settings.spark.delta_pip:
        from delta import configure_spark_with_delta_pip

        return configure_spark_with_delta_pip(
            builder, extra_packages=settings.spark.packages or None
        ).getOrCreate()
    if settings.spark.packages:
        builder = builder.config("spark.jars.packages", ",".join(settings.spark.packages))
    return builder.getOrCreate()
