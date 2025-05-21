import chess
from typing import Optional, Final
from .constants import *


class SimpleEval:

    currentBoard: chess.Board
    score: Optional[float]
    
    def __init__(self, board):
        self.currentBoard = board

    def basic_evaluate(self) -> float:
        self.score = 0
        if self.currentBoard.is_checkmate():
            if self.currentBoard.turn == chess.WHITE: self.score = -float('inf') 
            else: self.score = float('inf') 
        elif self.currentBoard.is_stalemate():
            self.score = 0.0
        else:
            for p in EVAL_PIECES:
                try:
                    val: float = PIECE_VALUES[p]
                except KeyError:
                    print("Key not found")
                    return 0     
                self.score += val * len(self.currentBoard.pieces(p, chess.WHITE))
                self.score -= val * len(self.currentBoard.pieces(p, chess.BLACK))
        return self.score
    