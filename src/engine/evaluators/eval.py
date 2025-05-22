import chess
from abc import ABC, abstractmethod
from typing import Optional


class Eval(ABC):
    """Abstract base class for chess position evaluators."""
    
    currentBoard: chess.Board
    score: Optional[float]
    
    def __init__(self, board: chess.Board):
        self.board = board
        self.score = None
    
    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate the current position and return a score.
        Positive values favor white, negative values favor black."""
        pass