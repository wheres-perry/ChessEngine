"""Fundamental constants shared by search and evaluation components."""

import os
from pathlib import Path
from typing import Final

from engine._core import moray_core as chess

DEFAULT_TIMEOUT: Final[float] = 250.0

DEFAULT_DEPTH: Final[int] = 6

DEFAULT_SYZYGY_PATH: Final[str] = os.environ.get(
    "SYZYGY_PATH", str(Path("data/syzygy"))
)
"""Path to Syzygy endgame tablebases, resolved via $SYZYGY_PATH or project default."""

PIECE_VALUES: Final[dict[int, float]] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

EVAL_PIECES: Final[set[int]] = {
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
}
