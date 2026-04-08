"""Evaluator protocol and evaluation base classes.

The native C++ evaluator base (``chess.evaluators.IEvaluator``) is the
actual runtime class used across the engine.  This module keeps the legacy
``Evaluator`` Protocol and ``EvalComponent`` alias for back-compat with
existing imports and type hints.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from engine._core import chess_engine_core as chess

# Re-export the C++ game-phase helper under the original name.
compute_game_phase = chess.evaluators.compute_game_phase


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


# Legacy alias — the C++ IEvaluator base is the canonical interface now.
EvalComponent = chess.evaluators.IEvaluator
