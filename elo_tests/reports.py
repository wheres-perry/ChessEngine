"""Reporting and output generation for Elo estimation runs.

This module provides reporters and writers for generating results
in JSONL, JSON, and Markdown formats.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from elo_tests.models import EloSummary


class JSONLReporter:
    """Reporter for appending game records to a JSONL file.

    Attributes:
        file_path: Path to the output JSONL file.

    """

    def __init__(self, file_path: Path) -> None:
        """Initialize the reporter with the output file path.

        Args:
            file_path: Path where JSONL records will be written.

        """
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: dict[str, object]) -> None:
        """Append a JSON object as a line to the output file.

        Args:
            payload: Dictionary to serialize and append.

        """
        with self.file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_summary_json(summary: EloSummary, out_path: Path) -> None:
    """Write the Elo summary to a JSON file.

    Args:
        summary: The EloSummary to write.
        out_path: Path for the output file.

    """
    out_path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def write_summary_markdown(summary: EloSummary, out_path: Path) -> None:
    """Write the Elo summary to a Markdown file.

    Args:
        summary: The EloSummary to write.
        out_path: Path for the output file.

    """
    text = "\n".join(
        [
            f"# Elo Run {summary.run_id}",
            "",
            f"- Candidate: {summary.candidate_id}",
            f"- Baseline: {summary.baseline_id}",
            f"- Seed: {summary.seed}",
            f"- Games: {summary.games_played}",
            f"- Pairs: {summary.pairs_played}",
            f"- Mean score: {summary.mean_score:.4f}",
            f"- Elo point estimate: {summary.elo_point:.2f}",
            f"- Primary CI ({summary.ci_method}, "
            f"{summary.ci_level:.2f}): [{summary.ci_primary_low:.2f}, "
            f"{summary.ci_primary_high:.2f}]",
            f"- Normal CI: [{summary.ci_normal_low:.2f}, {summary.ci_normal_high:.2f}]",
            f"- Bootstrap CI: [{summary.ci_bootstrap_low:.2f}, "
            f"{summary.ci_bootstrap_high:.2f}]",
            f"- Stop reason: {summary.stop_reason}",
            f"- Results JSONL: {summary.results_path}",
        ]
    )
    out_path.write_text(text + "\n", encoding="utf-8")
