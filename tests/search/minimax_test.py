"""
Tests for Minimax search algorithm focusing on config validation,
iterative deepening, time limiting, and terminal position handling.

These tests complement those in transposition_table_test.py and
zobrist_test.py by covering features not tested elsewhere.
"""

import logging
import time

import pytest

from engine._core import chess_engine_core as chess
from src.engine.config import EngineConfig, SearchConfig
from src.engine.evaluators import MockEvaluator
from src.engine.search.minimax import Minimax


class TestConfigValidation:
    """Test validation of EngineConfig during Minimax initialization."""

    def test_tt_aging_without_zobrist_raises(self):
        """Test that enabling TT aging without Zobrist hashing raises a ValueError."""
        # The validation happens in EngineConfig.__post_init__, not in Minimax
        # Since TT requires Zobrist, disabling Zobrist also requires disabling TT

        with pytest.raises(
            ValueError,
            match="TT aging requires Zobrist hashing",
        ):
            EngineConfig(
                minimax=SearchConfig(
                    use_zobrist=False,
                    use_transposition_table=False,
                    use_tt_aging=True,
                )
            )
        )
        resolver = DependencyResolver(config)

        with pytest.raises(DependencyResolutionError):
            resolver.resolve()


class TestPVSDependency:
    """Test that PVS is properly disabled when alpha-beta pruning is off."""

    def test_pvs_validation_error(self):
        """Test that enabling PVS without alpha-beta pruning raises an error."""
        with pytest.raises(
            ValueError,
            match=r"Principal Variation Search .* requires alpha-beta pruning",
        ):
            EngineConfig(
                minimax=SearchConfig(
                    use_alpha_beta=False,
                    use_pvs=True,
                    use_lmr=False,  # Turn off LMR since it requires alpha-beta
                )
            )
        )
        resolver = DependencyResolver(config)

        with pytest.raises(DependencyResolutionError):
            resolver.resolve()

    def test_pvs_with_alpha_beta_enabled(self):
        """Test that PVS works properly when alpha-beta is enabled."""
        cfg = EngineConfig(
            search=SearchConfig(
                use_alpha_beta=True,
                use_pvs=True,
            )
        )
        engine = Minimax(chess.Board(), MockEvaluator(chess.Board()), cfg)
        assert engine.use_pvs is True


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

        # Disable TT and related features since we're disabling Zobrist
        # (TT requires Zobrist)
        cfg = EngineConfig(
            search=SearchConfig(
                use_iddfs=True,
                use_zobrist=False,
                use_transposition_table=False,
                use_tt_aging=False,
                use_hash_move_ordering=False,  # Requires TT
                use_iid=False,  # Requires IDDFS + TT
                max_time=None,
            )
        )
        engine = Minimax(chess.Board(), MockEvaluator(chess.Board()), cfg)
        score, move = engine.find_top_move(depth=4)

        assert called == [1, 2, 3, 4]
        assert score == 4.0
        assert move == dummy_move


class TestTimeLimit:
    """Test time limit enforcement in search."""

    def test_check_time_limit_flags_time_up(self):
        """Test that _check_time_limit sets time_up flag when time is exceeded."""
        cfg = EngineConfig(
            search=SearchConfig(
                max_time=0.01,
            )
        )
        engine = Minimax(chess.Board(), MockEvaluator(chess.Board()), cfg)
        # Simulate search starting in the past
        engine.start_time = time.time() - 1.0
        assert engine._check_time_limit() is True
        assert engine.time_up is True
