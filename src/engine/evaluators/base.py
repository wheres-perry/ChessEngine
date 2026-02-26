"""Evaluator protocol and evaluation component interface.

Defines the public contract for all evaluators used by the search engine,
and the ``EvalComponent`` ABC for composable heuristic building-blocks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from engine._core import chess_engine_core as chess


# --- Public evaluator contract ---
@runtime_checkable
class Evaluator(Protocol):
    """Protocol every evaluator must satisfy.

    The search engine depends only on this interface, keeping the evaluator
    implementation completely swappable.
    """

    def go(self, board: chess.Board) -> float:
        """Return a heuristic score in **centipawns** (positive = White ahead)."""
        ...


# --- Game-phase helper ---
# Phase 1.0 = opening / early-middlegame, 0.0 = pure endgame.
# Based on non-pawn, non-king material.

_PHASE_WEIGHTS: dict[chess.PieceType, int] = {
    chess.QUEEN: 9,
    chess.ROOK: 5,
    chess.BISHOP: 3,
    chess.KNIGHT: 3,
}

# Max material at game start: 2Q(18) + 4R(20) + 4B(12) + 4N(12) = 62
_MAX_PHASE_MATERIAL: int = 62


def compute_game_phase(board: chess.Board) -> float:
    """Return game phase in [0.0, 1.0].

    1.0 -> opening / full-material middlegame.
    0.0 -> pure endgame (all major/minor pieces captured).
    """
    total = 0
    for piece_type, weight in _PHASE_WEIGHTS.items():
        for color in (chess.WHITE, chess.BLACK):
            total += len(board.pieces(piece_type, color)) * weight
    return min(1.0, total / _MAX_PHASE_MATERIAL)


# --- Evaluation component ABC ---
class EvalComponent(ABC):
    """A single, composable evaluation term.

    Each component returns a **centipawn** contribution for the position.
    Components receive the precomputed game phase so GSC-enabled components
    can blend opening/middlegame/endgame weights without recomputing it.
    """

    @abstractmethod
    def score(self, board: chess.Board, phase: float) -> float:
        """Return centipawn contribution (positive = White advantage)."""
