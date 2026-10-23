EXEC = docker compose exec workbench
SEASON ?= 2024

.PHONY: help up down cluster shell test test-all lint fmt typecheck ingest-results dagster jupyter app

help:  ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

up:  ## Build and start workbench + MLflow
	docker compose up -d --build

down:  ## Stop everything
	docker compose --profile cluster down

cluster:  ## Start the optional Spark standalone cluster
	docker compose --profile cluster up -d

shell:  ## Open a shell in the workbench
	$(EXEC) bash

test:  ## Fast unit tests
	$(EXEC) pytest

test-all:  ## Include slow tests
	$(EXEC) pytest -m ""

lint:  ## Lint
	$(EXEC) ruff check src tests

fmt:  ## Format
	$(EXEC) ruff format src tests
	$(EXEC) ruff check --fix src tests

typecheck:  ## Type-check
	$(EXEC) mypy src

ingest-results:  ## Ingest Jolpica results for SEASON into bronze
	$(EXEC) pitwall ingest jolpica --season $(SEASON) --endpoint results

dagster:  ## Dagster UI on :3000
	$(EXEC) dagster dev -h 0.0.0.0 -p 3000

jupyter:  ## JupyterLab on :8888
	$(EXEC) jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root

app:  ## Streamlit app on :8501
	$(EXEC) streamlit run app/streamlit_app.py --server.address 0.0.0.0
