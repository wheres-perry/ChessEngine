"""Tests for Syzygy endgame tablebase integration.

The test FEN ``K7/N7/k7/8/3p4/8/N7/8 w - - 0 1`` is a 5-piece position
(KNN vs Kp) where White has a forced mate when the 50-move drawing rule
is disabled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine._core import chess_engine_core as chess
from engine.config import EngineConfig, SearchConfig
from engine.constants import DEFAULT_SYZYGY_PATH
from engine.evaluators import SimpleEvaluator
from engine.search.minimax import Minimax
from engine.search.syzygy import SyzygyProber

_SYZYGY_DIR = Path(DEFAULT_SYZYGY_PATH)
_HAS_TABLES = _SYZYGY_DIR.is_dir() and any(_SYZYGY_DIR.glob("*.rtbw"))

# FEN: White King a8, Knight a7, Knight a2; Black King a6, Pawn d4.
# 5 pieces total — within 5-piece tablebase range.
WINNING_FEN = "K7/N7/k7/8/3p4/8/N7/8 w - - 0 1"


@pytest.mark.skipif(not _HAS_TABLES, reason="Syzygy tables not downloaded")
class TestSyzygyProber:
    """Tests for the SyzygyProber standalone class."""

    def test_probe_wdl_winning_position(self) -> None:
        """WDL probe reports a win for the winning side (50-move rule off)."""
        board = chess.Board.from_fen(WINNING_FEN)
        prober = SyzygyProber(str(_SYZYGY_DIR), use_50_move_rule=False)
        wdl = prober.probe_wdl(board)
        assert wdl is not None
        assert wdl > 0, f"Expected winning WDL, got {wdl}"

    def test_probe_dtz_winning_position(self) -> None:
        """DTZ probe returns a positive distance for the winning side."""
        board = chess.Board.from_fen(WINNING_FEN)
        prober = SyzygyProber(str(_SYZYGY_DIR), use_50_move_rule=False)
        dtz = prober.probe_dtz(board)
        assert dtz is not None
        assert dtz > 0, f"Expected positive DTZ, got {dtz}"

    def test_probe_returns_none_for_many_pieces(self) -> None:
        """Probe returns None for starting position (32 pieces)."""
        board = chess.Board()
        prober = SyzygyProber(str(_SYZYGY_DIR), use_50_move_rule=True)
        assert prober.probe_wdl(board) is None

    def test_piece_count(self) -> None:
        """Verify piece_count helper returns the expected count."""
        board = chess.Board.from_fen(WINNING_FEN)
        assert SyzygyProber.piece_count(board) == 5


@pytest.mark.skipif(not _HAS_TABLES, reason="Syzygy tables not downloaded")
class TestSyzygySearchIntegration:
    """Tests for Syzygy integration within the minimax search."""

    def test_engine_finds_winning_move_with_syzygy(self) -> None:
        """Engine finds a winning move with syzygy and 50-move rule disabled."""
        config = EngineConfig(
            search=SearchConfig(
                use_alpha_beta=True,
                use_pvs=True,
                use_move_ordering=True,
                use_transposition_table=True,
                use_tt_aging=True,
                use_hash_move_ordering=True,
                use_mvv_lva=True,
                use_killer_moves=True,
                use_history_heuristic=True,
                use_countermove_heuristic=True,
                use_quiescence_search=True,
                use_syzygy=True,
                syzygy_path=str(_SYZYGY_DIR),
                use_50_move_rule=False,
                max_time=30.0,
            ),
            search_depth=6,
        )
        board = chess.Board.from_fen(WINNING_FEN)
        evaluator = SimpleEvaluator()
        engine = Minimax(board, evaluator, config)

        score, move = engine.find_best_move(depth=4)

        assert score is not None
        assert move is not None
        assert score > 0, f"Expected winning score, got {score}"
