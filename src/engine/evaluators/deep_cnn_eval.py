from typing import Final, Optional

import chess
import torch
import torch.nn as nn
from src.engine.constants import *
from src.engine.evaluators.eval import Eval
from src.io_utils.to_tensor import create_tensor, NUM_PLANES


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and skip connections."""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Skip connection
        out = self.relu(out)
        return out


class DeepChessCNN(nn.Module):
    """Deep CNN architecture for chess position evaluation with residual connections."""
    
    def __init__(self, num_residual_blocks=8, base_channels=256):
        super(DeepChessCNN, self).__init__()
        
        # Initial convolution to expand channels
        self.initial_conv = nn.Conv2d(NUM_PLANES, base_channels, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(base_channels) for _ in range(num_residual_blocks)
        ])
        
        # Value head
        self.value_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Policy head (for future move prediction)
        self.policy_conv = nn.Conv2d(base_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096)  # Max possible moves
        
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
        
        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.relu(value)
        value = value.view(value.size(0), -1)  # Flatten
        value = self.dropout(value)
        value = self.relu(self.value_fc1(value))
        value = self.dropout(value)
        value = self.value_fc2(value)
        value = self.tanh(value)
        
        return value


class LightweightDeepCNN(nn.Module):
    """Lighter version of deep CNN for faster inference."""
    
    def __init__(self, num_residual_blocks=4, base_channels=128):
        super(LightweightDeepCNN, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(NUM_PLANES, base_channels, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(base_channels) for _ in range(num_residual_blocks)
        ])
        
        # Value head
        self.value_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
        
        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.relu(value)
        value = value.view(value.size(0), -1)  # Flatten
        value = self.dropout(value)
        value = self.relu(self.value_fc1(value))
        value = self.dropout(value)
        value = self.value_fc2(value)
        value = self.tanh(value)
        
        return value


class DeepCNN_Eval(Eval):
    """Deep CNN based chess position evaluator with modular architecture."""

    MAX_EVAL: Final[float] = 20.0  # Maximum evaluation score

    def __init__(
        self,
        board: chess.Board,
        model_path: Optional[str] = None,
        model_instance: Optional[nn.Module] = None,
        architecture: str = "deep",  # "deep", "lightweight"
        num_residual_blocks: int = 8,
        base_channels: int = 256,
    ):
        super().__init__(board)
        self.board = board
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.architecture = architecture
        
        if model_instance:
            self.model = model_instance
            self.model.to(self.device)
            self.model.eval()
        elif model_path:
            # Create model based on architecture
            if architecture == "deep":
                self.model = DeepChessCNN(num_residual_blocks, base_channels)
            elif architecture == "lightweight":
                self.model = LightweightDeepCNN(num_residual_blocks, base_channels)
            else:
                raise ValueError(f"Unknown architecture: {architecture}")
            
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except FileNotFoundError:
                print(f"Model file not found: {model_path}. Using random weights.")
            except Exception as e:
                print(f"Error loading model: {e}. Using random weights.")
            
            self.model.to(self.device)
            self.model.eval()
        else:
            # Create model with random weights
            if architecture == "deep":
                self.model = DeepChessCNN(num_residual_blocks, base_channels)
            elif architecture == "lightweight":
                self.model = LightweightDeepCNN(num_residual_blocks, base_channels)
            else:
                raise ValueError(f"Unknown architecture: {architecture}")
            
            self.model.to(self.device)
            self.model.eval()

    def load_model(self, model_path: str):
        """Load a trained model from the specified path."""
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.model.eval()
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def evaluate(self) -> float:
        """Evaluate the current position using the deep CNN.
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

    def get_model_info(self) -> dict:
        """Return information about the model architecture."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "architecture": self.architecture,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "model_class": self.model.__class__.__name__
        }
