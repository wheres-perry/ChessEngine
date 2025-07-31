import chess

from src.engine.constants import EVAL_PIECES, PIECE_VALUES
from src.engine.evaluators.evaluator import Evaluator


class SimpleEval(Evaluator):
    """
    A simple material-based chess position evaluator.

    This evaluator calculates a score based on piece values:
    - Positive scores favor White
    - Negative scores favor Black
    - Infinite scores represent checkmates
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
                piece_type: int = p
                try:
                    val: float = PIECE_VALUES[piece_type]
                except KeyError:
                    print("Key not found")
                    return 0
                self.score += val * len(self.board.pieces(piece_type, chess.WHITE))
                self.score -= val * len(self.board.pieces(piece_type, chess.BLACK))
        return self.score
