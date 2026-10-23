"""`pitwall` command-line entry point."""

from __future__ import annotations

import argparse
import logging

from pitwall import pipelines
from pitwall.silver.registry import SPECS

# Gold targets built per season; `gold-dims` rebuilds every dimension at once.
GOLD_TARGETS = {
    "fact_session_result": pipelines.build_fact_session_result,
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

    ff1 = ingest_sub.add_parser("fastf1")
    ff1.add_argument("--season", type=int, required=True)
    ff1.add_argument("--round", type=int, required=True)
    ff1.add_argument("--session", default="R")

    build = sub.add_parser("build", help="Build a silver or gold table")
    build.add_argument("table", choices=[*SPECS, "silver-laps", "gold-dims", *GOLD_TARGETS])
    build.add_argument("--season", type=int, help="Required for season-partitioned tables")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.command == "ingest" and args.source == "jolpica":
        if args.endpoint:
            pipelines.ingest_jolpica(args.season, args.endpoint, args.round)
        else:
            print(pipelines.ingest_jolpica_season(args.season))
    elif args.command == "ingest" and args.source == "fastf1":
        pipelines.ingest_fastf1_laps(args.season, args.round, args.session)
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
