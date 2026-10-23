# EMR Serverless custom image: EMR's Spark 3.5 + Delta, plus Python 3.11 and the pitwall
# package. Build from the repo root:
#   docker build -f deploy/aws/Dockerfile -t <ecr_repository_url>:latest .
FROM public.ecr.aws/emr-serverless/spark/emr-7.5.0:latest

USER root
# g++ lets PyTensor (PyMC, RQ1) compile its kernels instead of falling back to Python.
RUN dnf install -y python3.11 python3.11-pip gcc-c++ && dnf clean all

WORKDIR /opt/pitwall
COPY pyproject.toml README.md ./
COPY src ./src
COPY conf ./conf
# Result files some steps read (e.g. RQ2 effects for the race-state context).
COPY reports ./reports
# No `spark` extra: PySpark and Delta come from EMR, a pip copy would shadow them.
RUN python3.11 -m pip install --no-cache-dir -e ".[ds,causal]"

ENV PITWALL_ENV=aws \
    PITWALL_CONF_DIR=/opt/pitwall/conf \
    PYSPARK_PYTHON=/usr/bin/python3.11

# EMR Serverless runs jobs as this user.
USER hadoop:hadoop
