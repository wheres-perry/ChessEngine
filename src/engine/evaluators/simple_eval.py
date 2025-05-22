import chess
from typing import Optional, Final
from ..constants import *
from .eval import Eval


class SimpleEval(Eval):
    
    def __init__(self, board):
        super().__init__(board)

    def evaluate(self) -> float:
        """Implementation of the abstract method from Eval."""
        return self.basic_evaluate()
        
    def basic_evaluate(self) -> float:
        self.score = 0
        if self.board.is_checkmate():
            if self.board.turn == chess.WHITE: self.score = -float('inf') 
            else: self.score = float('inf') 
        elif self.board.is_stalemate():
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