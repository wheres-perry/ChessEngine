"""
Tests for Minimax search algorithm focusing on config validation,
iterative deepening, time limiting, and terminal position handling.
"""

from __future__ import annotations

import time

import pytest

from engine._core import chess_engine_core as chess
from src.engine.config import EngineConfig, SearchConfig
from src.engine.evaluators import MockEvaluator
from src.engine.module_dependency_resolver import (
    DependencyResolutionError,
    DependencyResolver,
)
from src.engine.search.minimax import Minimax


class TestConfigValidation:
    """Tests covering configuration validation helpers."""

    def test_tt_aging_requires_zobrist(self) -> None:
        config = EngineConfig()
        cfg = config.search
        cfg.use_transposition_table = False
        cfg.use_zobrist = False
        cfg.use_tt_aging = True

        resolver = DependencyResolver(config)
        with pytest.raises(
            DependencyResolutionError, match="TT aging requires Zobrist hashing"
        ):
            resolver.resolve()


class TestPVSDependency:
    """Test that PVS is properly disabled when alpha-beta pruning is off."""

    def test_pvs_validation_error(self) -> None:
        config = EngineConfig()
        cfg = config.search
        cfg.use_alpha_beta = False
        cfg.use_pvs = True
        # Disable other alpha-beta dependant features so the error pinpoints PVS
        cfg.use_quiescence_search = False
        cfg.use_null_move_pruning = False
        cfg.use_futility_pruning = False
        cfg.use_extended_futility_pruning = False
        cfg.use_reverse_futility_pruning = False
        cfg.use_aspiration_windows = False
        cfg.use_lmr = False

        resolver = DependencyResolver(config)
        with pytest.raises(
            DependencyResolutionError,
            match=r"Principal Variation Search.*alpha-beta",
        ):
            resolver.resolve()

    def test_pvs_with_alpha_beta_enabled(self) -> None:
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

    def test_iddfs_sequences_depths(self, monkeypatch) -> None:
        called: list[int] = []
        dummy_move = chess.Move.from_uci("a2a3")

        def fake_search(self, depth: int) -> tuple[float, chess.Move | None]:
            called.append(depth)
            return float(depth), (dummy_move if depth == 4 else None)

        monkeypatch.setattr(Minimax, "_search_fixed_depth", fake_search)

        cfg = EngineConfig(
            search=SearchConfig(
                use_iddfs=True,
                use_zobrist=False,
                use_transposition_table=False,
                use_tt_aging=False,
                use_hash_move_ordering=False,
                use_iid=False,
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

    def test_check_time_limit_flags_time_up(self) -> None:
        cfg = EngineConfig(
            search=SearchConfig(
                max_time=0.01,
            )
        )
        engine = Minimax(chess.Board(), MockEvaluator(chess.Board()), cfg)
        engine.start_time = time.time() - 1.0
        assert engine._check_time_limit() is True
        assert engine.time_up is True
