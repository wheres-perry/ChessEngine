"""
Tests for Minimax search algorithm focusing on config validation,
iterative deepening, time limiting, and terminal position handling.

These tests complement those in transposition_table_test.py and
zobrist_test.py by covering features not tested elsewhere.
"""

import logging
import time

import chess
import pytest

from src.engine.config import EngineConfig, MinimaxConfig
from src.engine.evaluators.mock_eval import MockEval
from src.engine.search.minimax import Minimax


class TestConfigValidation:
    """Test validation of EngineConfig during Minimax initialization."""

    def test_tt_aging_without_zobrist_raises(self):
        """Test that enabling TT aging without Zobrist hashing raises a ValueError."""
        # The validation happens in EngineConfig.__post_init__, not in Minimax

        with pytest.raises(
            ValueError,
            match="Transposition table aging requires Zobrist hashing to be enabled",
        ):
            EngineConfig(
                minimax=MinimaxConfig(
                    use_zobrist=False,
                    use_tt_aging=True,
                )
            )


class TestPVSDependency:
    """Test that PVS is properly disabled when alpha-beta pruning is off."""

    def test_pvs_disabled_with_warning(self, caplog):
        """Test that PVS is disabled when alpha-beta is disabled, with a warning."""
        caplog.set_level(logging.WARNING)
        cfg = EngineConfig(
            minimax=MinimaxConfig(
                use_alpha_beta=False,
                use_pvs=True,
            )
        )
        engine = Minimax(chess.Board(), MockEval(chess.Board()), cfg)
        assert engine.use_pvs is False
        assert any("Disabling PVS" in rec.getMessage() for rec in caplog.records)


class TestIterativeDeepening:
    """Test iterative deepening search implementation."""

    def test_iddfs_sequences_depths(self, monkeypatch):
        """Test that IDDFS calls _search_fixed_depth for each depth in sequence."""
        called = []
        dummy_move = chess.Move.from_uci("a2a3")

        def fake_search(self, depth):
            called.append(depth)
            # Return a move only on the final depth

            return float(depth), (dummy_move if depth == 4 else None)

        monkeypatch.setattr(Minimax, "_search_fixed_depth", fake_search)

        cfg = EngineConfig(
            minimax=MinimaxConfig(
                use_iddfs=True,
                use_zobrist=False,
                use_tt_aging=False,
                max_time=None,
            )
        )
        engine = Minimax(chess.Board(), MockEval(chess.Board()), cfg)
        score, move = engine.find_top_move(depth=4)

        assert called == [1, 2, 3, 4]
        assert score == 4.0
        assert move == dummy_move


class TestTimeLimit:
    """Test time limit enforcement in search."""

    def test_check_time_limit_flags_time_up(self):
        """Test that _check_time_limit sets time_up flag when time is exceeded."""
        cfg = EngineConfig(
            minimax=MinimaxConfig(
                max_time=0.01,
            )
        )
        engine = Minimax(chess.Board(), MockEval(chess.Board()), cfg)
        # Simulate search starting in the past

        engine.start_time = time.time() - 1.0
        assert engine._check_time_limit() is True
        assert engine.time_up is True
