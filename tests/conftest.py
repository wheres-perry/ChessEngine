"""Root conftest — shared fixtures, marker registration, and test hooks.

Markers registered here appear in ``pytest --markers`` and are enforced
by ``--strict-markers`` in pyproject.toml.
"""

from __future__ import annotations

import pytest

from engine._core import chess_engine_core as chess
from engine.config import EngineConfig, SearchConfig
from engine.evaluators import MockEvaluator


# ── Auto-tagging by directory ────────────────────────────────────────
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Automatically add markers based on test file location."""
    marker_map = {
        "benchmarks": pytest.mark.benchmark,
        "integration": pytest.mark.integration,
        "parity": pytest.mark.parity,
        "chess": pytest.mark.chess,
        "smoke": pytest.mark.slow,
    }
    for item in items:
        rel = str(item.fspath)
        for folder, marker in marker_map.items():
            if f"tests/{folder}/" in rel:
                item.add_marker(marker)


# ── Shared fixtures ──────────────────────────────────────────────────
@pytest.fixture
def board() -> chess.Board:
    """A fresh starting-position board."""
    return chess.Board()


@pytest.fixture
def mock_evaluator() -> MockEvaluator:
    """A zero-returning mock evaluator."""
    return MockEvaluator()


@pytest.fixture
def default_config() -> EngineConfig:
    """Sensible default ``EngineConfig`` for search tests."""
    return EngineConfig(
        search=SearchConfig(
            use_alpha_beta=True,
            use_move_ordering=True,
            use_transposition_table=True,
            max_time=None,
        )
    )


@pytest.fixture
def minimal_config() -> EngineConfig:
    """Bare-minimum config: alpha-beta only, no extras."""
    return EngineConfig(
        search=SearchConfig(
            use_alpha_beta=True,
            use_move_ordering=False,
            use_transposition_table=False,
            use_tt_aging=False,
            use_hash_move_ordering=False,
            use_iid=False,
            use_pvs=False,
            use_lmr=False,
            use_null_move_pruning=False,
            use_check_extensions=False,
            use_quiescence_search=False,
            use_killer_moves=False,
            use_history_heuristic=False,
            use_countermove_heuristic=False,
            use_mvv_lva=False,
            use_see_ordering=False,
            use_see_pruning_in_qs=False,
            use_delta_pruning=False,
            use_futility_pruning=False,
            use_extended_futility_pruning=False,
            use_reverse_futility_pruning=False,
            use_aspiration_windows=False,
            max_time=None,
        )
    )
