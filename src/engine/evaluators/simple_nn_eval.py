import chess
import torch
import torch.nn.functional as f
from torch import nn

from src.engine.board.halfkp_representation import board_to_input_tensor


# mypy: ignore-errors
# pyright: ignore
# pylint: skip-file
# ruff: noqa
class HalfKPNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden: nn.Linear = nn.Linear(1, 1)
        self.output: nn.Linear = nn.Linear(1, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class NeuralNetEvaluator:
    def __init__(self, board: chess.Board, model_path: str | None = None) -> None:
        self.board: chess.Board = board.copy()
        self.model: HalfKPNet = HalfKPNet()
        self.score: float | None = None
        self.model.eval()

    def evaluate(self) -> float:
        try:
            raise NotImplementedError()
        except Exception:
            return self._fallback_evaluation()

    def _fallback_evaluation(self) -> float:
        raise NotImplementedError()

    def update_board(self, board: chess.Board) -> None:
        raise NotImplementedError()
