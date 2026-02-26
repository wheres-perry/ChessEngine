"""Search test fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from engine.evaluators import MockEvaluator
from engine.search.minimax import Minimax

if TYPE_CHECKING:
    from engine._core import chess_engine_core as chess
    from engine.config import EngineConfig


@pytest.fixture
def search_engine(board: chess.Board, default_config: EngineConfig) -> Minimax:
    """A Minimax instance with sensible defaults, ready to search."""
    evaluator = MockEvaluator()
    return Minimax(board, evaluator, default_config)
