import chess
from src.engine.constants import *
from src.engine.evaluators.eval import *


class SimpleEval(Eval):
    """
    A simple material-based chess position evaluator.

    This evaluator calculates a score based on piece values:
    - Positive scores favor White
    - Negative scores favor Black
    - Infinite scores represent checkmate
    - Zero represents stalemate or equal material

    The evaluation considers:
    - Material balance (using piece values from constants)
    - Special positions (checkmate, stalemate)
    - Insufficient material draws

    It does not consider:
    - Piece positioning
    - Pawn structure
    - King safety
    - Control of center
    - Development
    """

    def __init__(self, board):
        super().__init__(board)

    def evaluate(self) -> float:
        self.score = 0
        if self.board.is_checkmate():
            if self.board.turn == chess.WHITE:
                self.score = -float("inf")
            else:
                self.score = float("inf")
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            self.score = 0.0
        else:
            for p in EVAL_PIECES:
                try:
                    val: float = PIECE_VALUES[p]
                except KeyError:
                    print("Key not found")
                    return 0
                self.score += val * len(self.board.pieces(p, chess.WHITE))
                self.score -= val * len(self.board.pieces(p, chess.BLACK))
        return self.score
