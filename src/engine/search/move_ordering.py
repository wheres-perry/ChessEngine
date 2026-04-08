"""Move ordering heuristics — re-exports the C++ implementation.

The real class lives in the compiled C++ extension ``engine._core``.  This
module preserves the legacy import path so callers keep working unchanged.
"""

from __future__ import annotations

from engine._core import chess_engine_core as chess

MoveSorter = chess.MoveSorter

# Legacy class-level piece value table used by engine.search.minimax helpers.
# Kept here (rather than in C++) because MoveSorter.PIECE_VALUES_CP is read
# with dict semantics (.get(piece_type, 0)) from several call sites.
MoveSorter.PIECE_VALUES_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}

__all__ = ["MoveSorter"]
