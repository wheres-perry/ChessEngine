"""Standalone Elo estimation package.

This package provides deterministic paired scheduling, simulated engine adapters,
JSONL raw game logging, Elo point estimation, confidence intervals, sequential
stopping, and CLI/config/reporting support for chess engine strength testing.
"""

from elo_tests.runner import run_elo_estimation

__all__ = ["run_elo_estimation"]
