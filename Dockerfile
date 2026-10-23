# Workbench image: Python + Java + PySpark. Also used for MLflow and the optional
# Spark standalone cluster, so driver and executors always share the same Python/libs.
FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps make git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[ds,dev]"

# Expose the pip-installed Spark distribution at a stable path for spark-class.
RUN ln -s "$(python -c 'import os, pyspark; print(os.path.dirname(pyspark.__file__))')" /opt/spark
ENV SPARK_HOME=/opt/spark \
    PYSPARK_PYTHON=python \
    PITWALL_ENV=local

CMD ["sleep", "infinity"]
