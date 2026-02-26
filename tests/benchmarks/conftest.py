"""Benchmark fixtures and persistent baseline configuration.

pytest-benchmark stores JSON results in ``.benchmarks/`` at the repo root.
Use ``--benchmark-autosave`` to persist each run and
``--benchmark-compare`` to diff against previous baselines.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.benchmarks.infrastructure import BaselineManager

BASELINES_DIR = Path(".benchmarks")


@pytest.fixture
def baseline_manager() -> BaselineManager:
    """Provide a ``BaselineManager`` rooted in the project ``.benchmarks/`` dir."""
    return BaselineManager(BASELINES_DIR)
