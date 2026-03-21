"""Tests for move ordering functionality.

This module contains unit tests for the MoveSorter class, including
hash move priority, killer moves, history heuristic, countermoves,
and MVV-LVA (Most Valuable Victim - Least Valuable Aggressor) ordering.
"""

from __future__ import annotations

from engine._core import chess_engine_core as chess
from engine.config import SearchConfig
from engine.search.move_ordering import MoveSorter


def _config(**overrides: object) -> SearchConfig:
    cfg = SearchConfig(
        use_move_ordering=True,
        use_hash_move_ordering=True,
        use_mvv_lva=True,
        use_see_ordering=True,
        use_killer_moves=True,
        use_history_heuristic=True,
        use_countermove_heuristic=True,
        use_alpha_beta=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_hash_move_is_ranked_first() -> None:
    """Verify the hash move is ranked first in the move list."""
    board = chess.Board()
    sorter = MoveSorter(_config())

    moves = list(board.generate_legal_moves())
    hash_move = chess.Move.from_uci("e2e4")

    ordered = sorter.sort_moves(
        board=board,
        moves=moves,
        ply=0,
        hash_move=hash_move,
        previous_move=None,
    )

    assert ordered[0] == hash_move


def test_on_beta_cutoff_updates_killer_and_history() -> None:
    """Verify beta cutoff updates killer moves and history tables."""
    board = chess.Board()
    sorter = MoveSorter(_config())
    move = chess.Move.from_uci("d2d4")

    sorter.on_beta_cutoff(
        move=move,
        ply=1,
        depth=4,
        previous_move=None,
        is_tactical=False,
    )

    assert move in sorter.killer_moves[1]
    assert (
        sorter.history_table[(move.from_square, move.to_square, int(move.promotion))]
        > 0
    )


def test_countermove_priority_after_update() -> None:
    """Verify countermove is prioritized after a beta cutoff update."""
    board = chess.Board()
    sorter = MoveSorter(_config())

    previous_move = chess.Move.from_uci("e7e5")
    countermove = chess.Move.from_uci("g1f3")

    sorter.on_beta_cutoff(
        move=countermove,
        ply=0,
        depth=3,
        previous_move=previous_move,
        is_tactical=False,
    )

    ordered = sorter.sort_moves(
        board=board,
        moves=list(board.generate_legal_moves()),
        ply=0,
        hash_move=None,
        previous_move=previous_move,
    )

    assert ordered[0] == countermove


def test_mvv_lva_prefers_queen_capture_over_pawn_capture() -> None:
    """Verify MVV-LVA prefers capturing a queen over capturing a pawn."""
    board = chess.Board.from_fen("7k/8/8/8/3Rq3/8/3p4/K7 w - - 0 1")
    sorter = MoveSorter(_config())

    captures = [m for m in board.generate_legal_moves() if board.is_capture(m)]
    ordered = sorter.sort_tactical(board, captures)

    assert ordered[0] == chess.Move.from_uci("d4e4")


def test_history_saturation_nonzero_after_updates() -> None:
    """Verify history saturation becomes nonzero after beta cutoffs."""
    sorter = MoveSorter(_config())

    sorter.on_beta_cutoff(
        move=chess.Move.from_uci("e2e4"),
        ply=0,
        depth=3,
        previous_move=None,
        is_tactical=False,
    )

    assert sorter.history_saturation() > 0.0
