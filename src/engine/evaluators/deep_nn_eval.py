from typing import Final, Optional

import chess
import torch
import torch.nn as nn
from src.engine.constants import *
from src.engine.evaluators.eval import Eval
from src.io_utils.to_tensor import create_tensor, NUM_PLANES


class DeepChessNN(nn.Module):
    def __init__(self):
        super(DeepChessNN, self).__init__()

        # Deeper CNN
        self.conv1 = nn.Conv2d(NUM_PLANES, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.tanh(x)
        return x


class DeepNN_Eval(Eval):
    """Deep neural network based chess position evaluator."""

    MAX_EVAL: Final[float] = 20.0  # Maximum evaluation score

    def __init__(
        self,
        board: chess.Board,
        model_path: Optional[str] = None,
        model_instance: Optional[nn.Module] = None,
    ):
        super().__init__(board)
        self.board = board
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_instance:
            self.model = model_instance
            self.model.to(self.device)
            self.model.eval()
        elif model_path:
            self.model = DeepChessNN()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            raise ValueError(
                "DeepNN_Eval requires either a model_path or a model_instance."
            )

    def evaluate(self) -> float:
        if self.board.is_checkmate():
            return -self.MAX_EVAL if self.board.turn else self.MAX_EVAL
        if (
            self.board.is_stalemate()
            or self.board.is_insufficient_material()
            or self.board.is_fifty_moves()
        ):
            return 0.0
        board_tensor = create_tensor(self.board)
        board_tensor = board_tensor.unsqueeze(0)
        board_tensor = board_tensor.to(self.device)
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(board_tensor)
        score = prediction.item() * self.MAX_EVAL
        return score
