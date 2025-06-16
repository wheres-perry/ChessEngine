from typing import Final

import chess
import torch
import torch.nn as nn
from src.engine.constants import *
from src.engine.evaluators.eval import Eval
from src.io_utils.to_tensor import create_tensor, NUM_PLANES


class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()

        # Input: 28 planes of 8x8 board representation

        self.conv1 = nn.Conv2d(NUM_PLANES, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        # Output scaled between -1 and 1, will be multiplied by MAX_EVAL

        x = self.tanh(x)
        return x


class NN_Eval(Eval):
    """Neural network based chess position evaluator."""

    MAX_EVAL: Final[float] = 20.0  # Maximum evaluation score

    def __init__(
        self,
        board: chess.Board,
        model_path: str | None = None,
        model_instance: nn.Module | None = None,
    ):
        super().__init__(board)
        self.board = board
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_instance:
            self.model = model_instance
            self.model.to(self.device)  # Ensure model is on the correct device
            self.model.eval()
        elif model_path:
            self.model = ChessNN()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            raise ValueError(
                "NN_Eval requires either a model_path or a model_instance."
            )

    def load_model(self, model_path: str):
        """Load a trained model from the specified path."""
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            print(f"Error loading model: {e}")

            # Initialize with random weights if loading fails

            pass

    def evaluate(self) -> float:
        """Evaluate the current position using the neural network.
        Returns a score where positive values favor white, negative values favor black.
        """
        if self.board.is_checkmate():
            return -self.MAX_EVAL if self.board.turn else self.MAX_EVAL
        if (
            self.board.is_stalemate()
            or self.board.is_insufficient_material()
            or self.board.is_fifty_moves()
        ):
            return 0.0
        # Convert board to tensor representation

        board_tensor = create_tensor(self.board)
        board_tensor = board_tensor.unsqueeze(0)  # Add batch dimension
        board_tensor = board_tensor.to(self.device)

        # Get model prediction

        with torch.no_grad():
            self.model.eval()
            prediction = self.model(board_tensor)
        # Scale the output from [-1, 1] to [-MAX_EVAL, MAX_EVAL]

        score = prediction.item() * self.MAX_EVAL

        return score
