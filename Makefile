EXEC = docker compose exec workbench
SEASON ?= 2024
SEASONS ?= 2018 2019 2020 2021 2022 2023 2024 2025
LAP_SEASONS ?= 2023 2024 2025

.PHONY: help up down cluster shell test test-all lint fmt typecheck lock ingest-results \
	backfill features rq1 rq2 rq3 research demo maintain dagster jupyter app

help:  ## Show targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

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

test-all:  ## Include slow tests (model parameter recovery)
	$(EXEC) pytest -m ""

lint:  ## Lint
	$(EXEC) ruff check src tests app

fmt:  ## Format
	$(EXEC) ruff format src tests app
	$(EXEC) ruff check --fix src tests app

typecheck:  ## Type-check
	$(EXEC) mypy src

lock:  ## Pin the resolved environment to requirements.lock
	$(EXEC) sh -c "pip freeze --exclude-editable | grep -v '^-e' > requirements.lock"

ingest-results:  ## Ingest + build Jolpica results for SEASON (bronze -> gold)
	$(EXEC) pitwall run $(SEASON)

backfill:  ## Results for SEASONS, then OpenF1 laps for LAP_SEASONS (~1 h, API rate limits)
	for s in $(SEASONS); do $(EXEC) pitwall run $$s || exit 1; done
	for s in $(LAP_SEASONS); do $(EXEC) pitwall ingest openf1 --season $$s && \
		$(EXEC) pitwall run $$s --no-ingest --laps || exit 1; done

features:  ## RQ1 features for LAP_SEASONS
	$(EXEC) pitwall research features rq1 $(foreach s,$(LAP_SEASONS),--season $(s))

rq1:  ## RQ1 tire degradation (baselines, PyMC, sensitivity) -> reports/rq1
	$(EXEC) pitwall research rq1

rq2:  ## RQ2 undercut (features need the RQ1 posterior) -> reports/rq2
	$(EXEC) pitwall research features rq2 $(foreach s,$(LAP_SEASONS),--season $(s))
	$(EXEC) pitwall research rq2

rq3:  ## RQ3 strategy simulator (backtest, decisions, scaling) -> reports/rq3
	$(EXEC) pitwall research rq3

research: features rq1 rq2 rq3  ## All research questions in order

demo:  ## Small end-to-end run: 2024 results, two races of laps, quick RQ1 fit
	$(EXEC) pitwall run 2024
	$(EXEC) pitwall ingest openf1 --season 2024 --round 1 --round 2
	$(EXEC) pitwall run 2024 --no-ingest --laps
	$(EXEC) pitwall research features rq1 --season 2024
	$(EXEC) pitwall research rq1 --train 2024 --test 2024 --draws 300 --tune 300 --chains 2 --no-sensitivity

maintain:  ## OPTIMIZE + Z-order + VACUUM all tables
	$(EXEC) pitwall maintain

dagster:  ## Dagster UI on :3000
	$(EXEC) dagster dev -h 0.0.0.0 -p 3000

jupyter:  ## JupyterLab on :8888
	$(EXEC) jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root

app:  ## Streamlit strategy explorer on :8501
	$(EXEC) streamlit run app/streamlit_app.py --server.address 0.0.0.0
