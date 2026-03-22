"""Engine adapters for Elo testing.

This package provides adapters for different chess engine types,
enabling uniform interaction during Elo estimation runs.
"""

from elo_tests.engines.factory import create_engine

__all__ = ["create_engine"]
