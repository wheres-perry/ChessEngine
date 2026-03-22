"""Command-line interface for the Elo estimation package."""

from __future__ import annotations

import argparse
import json

from elo_tests.config import load_run_config
from elo_tests.runner import run_elo_estimation


def main() -> None:
    """Run the Elo estimation CLI.

    Parses command-line arguments, loads the run configuration,
    executes the Elo estimation, and prints the summary as JSON.
    """
    parser = argparse.ArgumentParser(description="Standalone Elo estimator")
    parser.add_argument(
        "--config",
        default="elo_tests/configs/default.json",
        help="Path to JSON run config (optional)",
    )
    args = parser.parse_args()

    config = load_run_config(args.config)
    summary = run_elo_estimation(config)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
