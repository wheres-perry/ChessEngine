import chess
from src.engine.evaluators.eval import *


class MockEval(Eval):
    """
    A mock evaluator that returns a predefined score for testing.
    
    This evaluator provides a constant evaluation value:
    - Returns the same fixed score for any position
    - Ignores the actual board state
    - Can be updated during test execution with set_score()
    
    The evaluation provides:
    - Controllable, predictable outputs for unit tests
    - Isolation from actual evaluation logic
    - Consistent results regardless of position
    
    It does not consider:
    - Any aspect of the chess position
    - Material balance
    - Game state (checkmate, stalemate, etc.)
    """

    def __init__(self, board: chess.Board, fixed_score: float = 0.0):
        super().__init__(board)
        self.fixed_score = fixed_score
        self.score = fixed_score

    def evaluate(self) -> float:
        self.score = self.fixed_score
        return self.score

    def set_score(self, score: float):
        """
        Allows updating the fixed score after instantiation.
        """
        self.fixed_score = score
        self.score
