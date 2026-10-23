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

    sub.add_parser("fingerprints", help="Row counts + checksums of silver and gold tables")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.command == "run":
        pipelines.run_season(args.season, ingest=not args.no_ingest)
        if args.laps:
            pipelines.run_season_laps(
                args.season, ingest=not args.no_ingest, telemetry=args.telemetry
            )
        return
    if args.command == "maintain":
        print(json.dumps(pipelines.maintain(), indent=2))
        return
    if args.command == "fingerprints":
        print(json.dumps(pipelines.fingerprints(), indent=2, sort_keys=True))
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
