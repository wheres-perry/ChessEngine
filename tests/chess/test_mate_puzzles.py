"""Chess puzzle tests — mate-in-N, tactics, correctness.

These tests verify that the search engine finds known-correct moves
for curated puzzle positions.  Add new puzzle sets by creating a
``.fen`` file in ``tests/chess/data/`` and loading it via the helpers
in ``conftest.py``.
"""

from __future__ import annotations

import pytest

from engine._core import chess_engine_core as chess
from engine.config import EngineConfig, SearchConfig
from engine.evaluators import SimpleEvaluator
from engine.search.minimax import Minimax
from tests.chess.conftest import load_fen_file


def _find_move(fen: str, depth: int) -> str | None:
    """Run the engine on *fen* at *depth* and return the best move in UCI."""
    board = chess.Board.from_fen(fen)
    evaluator = SimpleEvaluator()
    config = EngineConfig(
        search=SearchConfig(
            use_alpha_beta=True,
            use_move_ordering=True,
            use_transposition_table=True,
            use_quiescence_search=True,
            max_time=5.0,
        ),
    )
    engine = Minimax(board, evaluator, config)
    _score, move = engine.find_top_move(depth=depth)
    return move.uci() if move else None


# ── Mate-in-1 ────────────────────────────────────────────────────────
class TestMateIn1:
    """Engine must find forced mate in 1 move."""

    PUZZLES = load_fen_file("mate_in_1.fen")

    @pytest.mark.parametrize(
        ("fen", "expected"),
        PUZZLES,
        ids=[f"puzzle_{i}" for i in range(len(PUZZLES))],
    )
    def test_finds_mate(self, fen: str, expected: str) -> None:
        found = _find_move(fen, depth=2)  # depth 2 to see the mating reply
        assert found == expected, f"Expected {expected}, got {found} for FEN: {fen}"


# ── Placeholder for future puzzle categories ─────────────────────────
# class TestMateIn2:
#     """Engine must find forced mate in 2 moves."""
#     PUZZLES = load_fen_file("mate_in_2.fen")
#     ...
#
# class TestTactics:
#     """Engine must find winning tactical shots."""
#     PUZZLES = load_fen_file("tactics.fen")
#     ...
