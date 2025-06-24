from typing import Final

import chess

DEFAULT_TIMEOUT: Final[float] = 2500.0

DEFAULT_DEPTH: Final[int] = 7

PIECE_VALUES: Final[dict[int, float]] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

EVAL_PIECES: Final[set] = set(
    [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
)
