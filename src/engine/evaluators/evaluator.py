from abc import ABC, abstractmethod

import chess


class Evaluator(ABC):
    """Abstract base class for chess position evaluators."""

    current_board: chess.Board
    score: float | None

    def __init__(self, board: chess.Board):
        self.board = board
        self.score = None

    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate the current position and return a score.
        Positive values favor white, negative values favor black."""
