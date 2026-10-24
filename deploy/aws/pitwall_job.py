"""EMR Serverless entry point: the arguments are a `pitwall` command line.

run_job.sh run 2025 --laps   ==   pitwall run 2025 --laps
"""

import sys

from pitwall.cli import main

if __name__ == "__main__":
    main(sys.argv[1:])
