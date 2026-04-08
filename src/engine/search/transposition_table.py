"""Transposition table — re-exports the C++ implementation.

The real class lives in the compiled C++ extension ``engine._core``.  This
module preserves the legacy import path so callers keep working unchanged.
"""

from __future__ import annotations

from typing import Literal

from engine._core import chess_engine_core as chess

BoundType = Literal["exact", "lower", "upper"]

TranspositionTable = chess.TranspositionTable
TTEntry = chess.TTEntry

__all__ = ["BoundType", "TTEntry", "TranspositionTable"]
