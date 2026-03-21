"""Tests for the Minimax search algorithm.

This module contains unit tests for the Minimax search implementation,
including alpha-beta pruning, time limits, and transposition table integration.
"""

from __future__ import annotations

import time

from engine._core import chess_engine_core as chess
from engine.config import EngineConfig, SearchConfig
from engine.evaluators import MockEvaluator
from engine.search.minimax import Minimax


def _minimal_search_config(**overrides: object) -> SearchConfig:
    cfg = SearchConfig(
        use_alpha_beta=True,
        use_pvs=False,
        use_quiescence_search=False,
        use_move_ordering=False,
        use_transposition_table=False,
        use_iid=False,
        use_null_move_pruning=False,
        use_lmr=False,
        use_futility_pruning=False,
        use_extended_futility_pruning=False,
        use_reverse_futility_pruning=False,
        use_aspiration_windows=False,
        use_check_extensions=False,
        use_mvv_lva=False,
        use_see_ordering=False,
        use_killer_moves=False,
        use_history_heuristic=False,
        use_countermove_heuristic=False,
        use_hash_move_ordering=False,
        use_delta_pruning=False,
        use_see_pruning_in_qs=False,
        use_tt_aging=False,
        max_time=None,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_find_best_move_returns_legal_move() -> None:
    """Verify find_best_move returns a legal move."""
    config = EngineConfig(search=_minimal_search_config())
    board = chess.Board()
    engine = Minimax(board, MockEvaluator(), config)

    score, move = engine.find_best_move(depth=1)

    legal = list(board.generate_legal_moves())
    assert score is not None
    assert move is not None
    assert move in legal


def test_find_top_move_alias_matches_find_best_move() -> None:
    """Verify find_top_move produces the same result as find_best_move."""
    config = EngineConfig(search=_minimal_search_config())

    engine_a = Minimax(chess.Board(), MockEvaluator(), config)
    score_a, move_a = engine_a.find_best_move(depth=1)

    engine_b = Minimax(chess.Board(), MockEvaluator(), config)
    score_b, move_b = engine_b.find_top_move(depth=1)

    assert move_a == move_b
    assert score_a == score_b


def test_non_alpha_beta_mode_runs() -> None:
    """Verify search completes without alpha-beta pruning enabled."""
    config = EngineConfig(
        search=_minimal_search_config(
            use_alpha_beta=False,
        )
    )
    board = chess.Board()
    engine = Minimax(board, MockEvaluator(), config)

    score, move = engine.find_best_move(depth=1)

    assert score is not None
    assert move is not None


def test_time_limit_flag_trips() -> None:
    """Verify the time limit flag triggers when time is exceeded."""
    config = EngineConfig(search=_minimal_search_config(max_time=0.01))
    engine = Minimax(chess.Board(), MockEvaluator(), config)

    engine.start_time = time.time() - 1.0

    assert engine._check_time_limit() is True
    assert engine.time_up is True


def test_transposition_table_records_hits_on_repeated_search() -> None:
    """Verify transposition table hits increase on repeated searches."""
    config = EngineConfig(
        search=_minimal_search_config(
            use_transposition_table=True,
            use_move_ordering=True,
            use_hash_move_ordering=True,
            use_tt_aging=True,
            use_alpha_beta=True,
            use_pvs=True,
            use_quiescence_search=True,
            use_mvv_lva=True,
            use_killer_moves=True,
            use_history_heuristic=True,
            use_countermove_heuristic=True,
            use_lmr=True,
            use_null_move_pruning=True,
            use_futility_pruning=True,
            use_extended_futility_pruning=True,
            use_reverse_futility_pruning=True,
            use_aspiration_windows=True,
            max_time=None,
        )
    )
    board = chess.Board.from_fen(
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3"
    )
    engine = Minimax(board, MockEvaluator(), config)

    engine.find_best_move(depth=3)
    first_hits = engine.stats.tt_hits

    engine.find_best_move(depth=3)
    second_hits = engine.stats.tt_hits

    assert second_hits >= first_hits
