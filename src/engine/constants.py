from typing import Final
import chess


PIECE_VALUES: Final[dict[int, float]] = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
}

EVAL_PIECES: Final[set] = set([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])