"""`pitwall` command-line entry point."""

from __future__ import annotations

import argparse
import json
import logging

from pitwall import pipelines
from pitwall.silver.registry import SPECS

# Gold targets built per season; `gold-dims` rebuilds every dimension at once.
GOLD_TARGETS = {
    "fact_session_result": pipelines.build_fact_session_result,
    "driver_id_map": pipelines.build_driver_id_map,
    "fact_lap": pipelines.build_fact_lap,
    "fact_stint": pipelines.build_fact_stint,
    "fact_pit_stop": pipelines.build_fact_pit_stop,
    "fact_telemetry_sample": pipelines.build_fact_telemetry_sample,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pitwall")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Load a source into bronze")
    ingest_sub = ingest.add_subparsers(dest="source", required=True)

    jolpica = ingest_sub.add_parser("jolpica")
    jolpica.add_argument("--season", type=int, required=True)
    jolpica.add_argument("--endpoint", help="One endpoint; omit to load the whole season")
    jolpica.add_argument("--round", type=int)

    of1 = ingest_sub.add_parser("openf1")
    of1.add_argument("--season", type=int, required=True)
    of1.add_argument("--round", type=int, action="append", help="Repeatable; omit for all")
    of1.add_argument("--telemetry", action="store_true", help="Also load car + location data")

    ff1 = ingest_sub.add_parser("fastf1")
    ff1.add_argument("--season", type=int, required=True)
    ff1.add_argument("--round", type=int, help="Omit to load every round of the season")
    ff1.add_argument("--session", default="R")
    ff1.add_argument("--telemetry", action="store_true", help="Also load car + position data")

    build = sub.add_parser("build", help="Build a silver or gold table")
    build.add_argument("table", choices=[*SPECS, "silver-laps", "gold-dims", *GOLD_TARGETS])
    build.add_argument("--season", type=int, help="Required for season-partitioned tables")

    run = sub.add_parser("run", help="Run the pipeline for a season without Dagster")
    run.add_argument("season", type=int)
    run.add_argument("--no-ingest", action="store_true", help="Rebuild from existing bronze")
    run.add_argument("--laps", action="store_true", help="Also run the FastF1 lap graph")
    run.add_argument("--telemetry", action="store_true", help="With --laps: include telemetry")

    sub.add_parser("maintain", help="OPTIMIZE (+Z-order) and VACUUM every lakehouse table")

    fp = sub.add_parser("fingerprints", help="Row counts + checksums of silver and gold tables")
    fp.add_argument("--season", type=int, help="Only season-partitioned tables, this season")
    cp = sub.add_parser("copy", help="Copy Delta tables between lakehouses (e.g. local -> S3)")
    cp.add_argument("--from", dest="source", required=True, help="Source lakehouse root")
    cp.add_argument("--to", dest="target", default=None, help="Default: this env's root")
    cp.add_argument("--layer", action="append", default=None, help="Repeatable; default bronze")
    cp.add_argument("--season", type=int, action="append", help="Repeatable; default all")
    cp.add_argument("--exclude", action="append", default=[], help="Table name to skip")
    sub.add_parser("perf", help="Layout before/after OPTIMIZE + skew -> reports/engineering")

    rq = sub.add_parser("research", help="Feature tables and research-question runs")
    rq_sub = rq.add_subparsers(dest="target", required=True)
    feat = rq_sub.add_parser("features", help="Build a feature table for seasons")
    feat.add_argument("table", choices=["rq1", "rq2", "rq3", "rq4"])
    feat.add_argument("--season", type=int, action="append", required=True)
    rq1 = rq_sub.add_parser("rq1", help="Tire degradation: baselines, PyMC model, checks")
    rq1.add_argument("--train", type=int, nargs="+", default=[2023, 2024])
    rq1.add_argument("--test", type=int, default=2025)
    rq1.add_argument("--draws", type=int, default=2000)
    rq1.add_argument("--tune", type=int, default=2000)
    rq1.add_argument("--chains", type=int, default=4)
    rq1.add_argument("--seed", type=int, default=42)
    rq1.add_argument("--no-sensitivity", action="store_true")
    rq2 = rq_sub.add_parser("rq2", help="Undercut: naive, IPW, DML, diagnostics, refutations")
    rq2.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    rq2.add_argument("--seed", type=int, default=42)
    eda = rq_sub.add_parser("eda", help="Coverage audit and distributions -> reports/eda")
    eda.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    rq3 = rq_sub.add_parser("rq3", help="Strategy simulator: backtest, decisions, scaling")
    rq3.add_argument("--season", type=int, default=2025)
    rq3.add_argument("--backtest-sims", type=int, default=10_000)
    rq3.add_argument("--decision-sims", type=int, default=800)
    rq3.add_argument("--benchmark-sims", type=int, default=200_000)
    rq3.add_argument("--no-million", action="store_true")
    rq3.add_argument("--seed", type=int, default=42)
    rq4 = rq_sub.add_parser("rq4", help="Calibrated pre-race P(podium) / P(points)")
    rq4.add_argument("--eval", type=int, nargs=2, default=[2022, 2025], metavar=("FROM", "TO"))
    rq4.add_argument("--simulator-sims", type=int, default=10_000)
    rq4.add_argument("--no-simulator", action="store_true", help="Skip the RQ3 comparison")
    rq4.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.command == "run":
        pipelines.run_season(args.season, ingest=not args.no_ingest)
        if args.laps:
            pipelines.run_season_laps(
                args.season, ingest=not args.no_ingest, telemetry=args.telemetry
            )
        return
    if args.command == "research":
        from pitwall import research

        if args.target == "features":
            build = {
                "rq1": research.build_feat_rq1,
                "rq2": research.build_feat_rq2,
                "rq3": research.build_feat_rq3,
                "rq4": research.build_feat_rq4,
            }[args.table]
            print({s: build(s) for s in args.season})
        elif args.target == "rq1":
            results = research.run_rq1(
                args.train,
                args.test,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                seed=args.seed,
                sensitivity=not args.no_sensitivity,
            )
            print(json.dumps({k: results[k] for k in ("b0", "b1", "bayes")}, indent=2))
        elif args.target == "rq2":
            results = research.run_rq2(args.seasons, seed=args.seed)
            print(json.dumps(results["cycle"]["primary"], indent=2, default=str))
        elif args.target == "eda":
            print(json.dumps(research.run_eda(args.seasons), indent=2, default=str))
        elif args.target == "rq3":
            results = research.run_rq3(
                args.season,
                backtest_sims=args.backtest_sims,
                decision_sims=args.decision_sims,
                benchmark_sims=args.benchmark_sims,
                million_run=not args.no_million,
                seed=args.seed,
            )
            print(json.dumps({k: results[k] for k in ("backtest", "decision")}, default=str))
        elif args.target == "rq4":
            results = research.run_rq4(
                (args.eval[0], args.eval[1]),
                simulator_season=None if args.no_simulator else args.eval[1],
                simulator_sims=args.simulator_sims,
                seed=args.seed,
            )
            print(json.dumps(results["eval"], indent=2))
        return
    if args.command == "maintain":
        print(json.dumps(pipelines.maintain(), indent=2))
        return
    if args.command == "perf":
        from pitwall import perf

        print(json.dumps(perf.layout_report(), indent=2, default=str))
        return
    if args.command == "fingerprints":
        print(json.dumps(pipelines.fingerprints(season=args.season), indent=2, sort_keys=True))
        return
    if args.command == "copy":
        from pitwall.config import get_settings

        copied = pipelines.copy_tables(
            args.source,
            args.target or get_settings().lakehouse_root,
            tuple(args.layer or ["bronze"]),
            args.season,
            tuple(args.exclude),
        )
        print(json.dumps(copied, indent=2, sort_keys=True))
        return

    if args.command == "ingest" and args.source == "jolpica":
        if args.endpoint:
            pipelines.ingest_jolpica(args.season, args.endpoint, args.round)
        else:
            print(pipelines.ingest_jolpica_season(args.season))
    elif args.command == "ingest" and args.source == "openf1":
        print(pipelines.ingest_openf1_season(args.season, args.round, args.telemetry))
    elif args.command == "ingest" and args.source == "fastf1":
        if args.round is None:
            pipelines.ingest_fastf1_season(args.season, [args.session], args.telemetry)
        else:
            print(
                pipelines.ingest_fastf1_session(
                    args.season, args.round, args.session, args.telemetry
                )
            )
    elif args.command == "build" and args.table == "gold-dims":
        print(pipelines.build_gold_dims())
    elif args.command == "build":
        if args.season is None:
            parser.error(f"{args.table} needs --season")
        if args.table in GOLD_TARGETS:
            print(GOLD_TARGETS[args.table](args.season))
        else:
            name = "laps" if args.table == "silver-laps" else args.table
            print(pipelines.build_silver(name, args.season))


if __name__ == "__main__":
    main()
