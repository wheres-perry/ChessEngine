"""
Neural Network evaluator for chess positions.

Uses PyTorch to evaluate positions via a trained neural network.
Can optionally use handcoded features as inputs (hybrid approach).

Tree 2: State Evaluation Optimizations
Supports modular feature extraction for NN input features.
"""

from typing import Any

import numpy as np

from engine._core import chess_engine_core as chess
from src.engine.config import EvaluationConfig
from src.engine.evaluators.base_evaluator import BaseEvaluator

# Conditional torch import, avoids circular import
try:
    import torch

    TORCH_AVAILABLE: bool = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


class NeuralNetworkEvaluator(BaseEvaluator):
    """
    Neural network-based evaluator.

    Can operate in several modes:
    1. Pure NN: Board representation -> NN -> evaluation
    2. Hybrid: Board + handcoded features -> NN -> evaluation
    3. Feature-based: Use config flags to select which features to extract

    The neural network architecture and weights should be loaded separately.
    """

    def __init__(
        self,
        board: chess.Board,
        config: EvaluationConfig,
        model: Any | None = None,
        model_path: str | None = None,
    ):
        """
        Initialize neural network evaluator.

        Args:
            board: Chess board to evaluate
            config: Configuration specifying features to use
            model: Pre-loaded PyTorch model (optional)
            model_path: Path to model weights file (optional)
        """
        super().__init__(board, config)

        self.model = model
        self.model_path = model_path

        # Will be set to True once PyTorch is available
        self.pytorch_available = self._check_pytorch()

        if self.pytorch_available and model is None and model_path:
            self._load_model(model_path)

    def _check_pytorch(self) -> bool:
        """Check if PyTorch is available."""
        return TORCH_AVAILABLE

    def _load_model(self, model_path: str) -> None:
        """Load a trained PyTorch model from file."""
        if not self.pytorch_available or torch is None:
            raise ImportError(
                "PyTorch is not installed. Install with: pip install torch"
            )

        try:
            self.model = torch.load(model_path)
            self.model.eval()  # Set to evaluation mode
        except (OSError, RuntimeError) as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e

    def evaluate(self) -> float:
        """
        Evaluate position using neural network.

        Returns:
            Evaluation score (positive favors White)
        """
        if not self.pytorch_available or self.model is None:
            # Fallback to simple material count if NN not available
            if self.config.use_material:
                white_material, black_material = self.count_material()
                return (white_material - black_material) / 100.0
            return 0.0

        # Extract features based on configuration
        features = self._extract_features()

        # Run through neural network
        # torch is guaranteed to be available here since we checked above
        assert torch is not None, "PyTorch should be available at this point"

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            output = self.model(features_tensor)
            evaluation: float = output.item()
            return evaluation

    def _extract_features(self) -> np.ndarray:
        """Extract modular feature vector respecting Tree 2 dependencies."""
        features = []

        # Always include basic board representation
        # 12 planes (6 piece types x 2 colors) x 64 squares = 768 features
        features.extend(self._encode_board_planes())

        # Node E0: Material features (if enabled)
        if self.config.use_material:
            features.extend(self._encode_material_features())

        # Node E4: Mobility features (if enabled and PST enabled)
        if self.config.use_mobility and self.config.use_pst:
            features.extend(self._encode_mobility_features())

        # Node E3: Pawn structure (if enabled and PST enabled)
        if self.config.use_pawn_structure and self.config.use_pst:
            features.extend(self._encode_pawn_features())

        # Node E5: King safety (if enabled and mobility enabled)
        if self.config.use_king_safety and self.config.use_mobility:
            features.extend(self._encode_king_safety_features())

        # Game phase feature (useful for E2 tapered evaluation)
        if self.config.use_tapered_eval:
            features.append(self.get_game_phase())

        return np.array(features, dtype=np.float32)

    def _encode_board_planes(self) -> list[float]:
        """Encode board as 12 binary planes (768 values)."""
        planes = []

        piece_types = [
            chess.PAWN,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
            chess.KING,
        ]
        colors = [chess.WHITE, chess.BLACK]

        for color in colors:
            for piece_type in piece_types:
                plane = [0.0] * 64
                pieces = self.board.pieces(piece_type, color)
                for square in pieces:
                    plane[square] = 1.0
                planes.extend(plane)

        return planes

    def _encode_material_features(self) -> list[float]:
        """Encode normalized material count features (10 values)."""
        features = []

        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in piece_types:
                count = self.get_piece_count(piece_type, color)
                # Normalize by typical max count
                max_count = (
                    8
                    if piece_type == chess.PAWN
                    else (
                        2
                        if piece_type in [chess.ROOK, chess.BISHOP, chess.KNIGHT]
                        else 1
                    )
                )
                features.append(count / max_count)

        return features

    def _encode_mobility_features(self) -> list[float]:
        """Encode normalized mobility features (2 values)."""
        white_mobility = 0
        black_mobility = 0

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue

            mobility = self.get_piece_mobility(square)

            if piece.color == chess.WHITE:
                white_mobility += mobility
            else:
                black_mobility += mobility

        # Normalize (typical total mobility is ~20-40 moves per side)
        return [white_mobility / 50.0, black_mobility / 50.0]

    def _encode_pawn_features(self) -> list[float]:
        """Encode pawn structure features (doubled, isolated, passed)."""
        features = []

        for color in [chess.WHITE, chess.BLACK]:
            pawns = self.board.pieces(chess.PAWN, color)

            # Count doubled, isolated, and passed pawns
            doubled = 0
            isolated = 0
            passed = 0

            for pawn_square in pawns:
                file = chess.square_file(pawn_square)

                # Check doubled
                pawns_on_file = [p for p in pawns if chess.square_file(p) == file]
                if len(pawns_on_file) > 1:
                    doubled += 1

                # Check isolated
                has_neighbor = any(
                    chess.square_file(p) in [file - 1, file + 1] for p in pawns
                )
                if not has_neighbor:
                    isolated += 1

                # TODO: Check passed (simplified)
                # passed += 1 if is_passed else 0

            features.extend([doubled / 8.0, isolated / 8.0, passed / 8.0])

        return features

    def _encode_king_safety_features(self) -> list[float]:
        """Encode king safety features (pawn shield and exposure)."""
        features = []

        for color in [chess.WHITE, chess.BLACK]:
            king_square = self.board.king(color)
            if king_square is None:
                features.extend([0.0, 0.0])
                continue

            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)

            # Count pawn shield
            pawn_shield = 0
            for file_offset in [-1, 0, 1]:
                check_file = king_file + file_offset
                if not (0 <= check_file <= 7):
                    continue

                pawn_rank = king_rank + (1 if color == chess.WHITE else -1)
                if 0 <= pawn_rank <= 7:
                    check_square = chess.square(check_file, pawn_rank)
                    piece = self.board.piece_at(check_square)
                    if (
                        piece
                        and piece.piece_type == chess.PAWN
                        and piece.color == color
                    ):
                        pawn_shield += 1

            # Normalize
            features.append(pawn_shield / 3.0)

            # King exposure (0-7 for rank, normalized)
            exposure = king_rank if color == chess.WHITE else (7 - king_rank)
            features.append(exposure / 7.0)

        return features
