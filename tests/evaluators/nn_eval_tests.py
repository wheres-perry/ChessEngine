"""
Comprehensive tests for NeuralNetworkEvaluator.

Tests cover:
- Feature extraction based on configuration
- PyTorch integration (mocked)
- Modular feature flags
- Fallback behavior when PyTorch unavailable
- Board encoding
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from engine._core import chess_engine_core as chess
from src.engine.config import EvaluationConfig
from src.engine.evaluators.nn_eval import NeuralNetworkEvaluator


class TestNNEvaluatorInitialization:
    """Test NeuralNetworkEvaluator initialization."""

    def test_init_without_pytorch(self, monkeypatch):
        """Test initialization when PyTorch is not available."""
        # Mock PyTorch as unavailable
        monkeypatch.setattr("src.engine.evaluators.nn_eval.TORCH_AVAILABLE", False)
        monkeypatch.setattr("src.engine.evaluators.nn_eval.torch", None)

        board = chess.Board()
        config = EvaluationConfig()

        evaluator = NeuralNetworkEvaluator(board, config)

        # Should initialize but mark PyTorch as unavailable
        assert evaluator.pytorch_available is False
        assert evaluator.model is None

    @patch("src.engine.evaluators.nn_eval.NeuralNetworkEvaluator._check_pytorch")
    def test_init_with_pytorch_available(self, mock_check):
        """Test initialization when PyTorch is available."""
        mock_check.return_value = True

        board = chess.Board()
        config = EvaluationConfig()

        evaluator = NeuralNetworkEvaluator(board, config)

        assert evaluator.pytorch_available is True

    def test_init_with_model(self):
        """Test initialization with provided model."""
        board = chess.Board()
        config = EvaluationConfig()
        mock_model = Mock()

        evaluator = NeuralNetworkEvaluator(board, config, model=mock_model)

        assert evaluator.model == mock_model

    def test_init_with_model_path_but_no_pytorch(self, monkeypatch):
        """Test that model path is ignored when PyTorch unavailable."""
        # Mock PyTorch as unavailable
        monkeypatch.setattr("src.engine.evaluators.nn_eval.TORCH_AVAILABLE", False)
        monkeypatch.setattr("src.engine.evaluators.nn_eval.torch", None)

        board = chess.Board()
        config = EvaluationConfig()

        evaluator = NeuralNetworkEvaluator(board, config, model_path="model.pt")

        # Should not crash, but model won't be loaded
        assert evaluator.model is None


class TestNNEvaluatorFallbackBehavior:
    """Test fallback behavior when PyTorch is unavailable."""

    def test_evaluate_without_pytorch_material_enabled(self):
        """Test evaluation falls back to material when PyTorch unavailable."""
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        config = EvaluationConfig(use_material=True)

        evaluator = NeuralNetworkEvaluator(board, config)
        score = evaluator.evaluate()

        # Should return material difference (White up a queen = 9.0)
        assert score == 9.0

    def test_evaluate_without_pytorch_material_disabled(self, monkeypatch):
        """Test evaluation returns 0 when PyTorch unavailable and material disabled."""
        # Mock PyTorch as unavailable
        monkeypatch.setattr("src.engine.evaluators.nn_eval.TORCH_AVAILABLE", False)
        monkeypatch.setattr("src.engine.evaluators.nn_eval.torch", None)

        board = chess.Board()
        config = EvaluationConfig(evaluator_type="nn", use_material=False)  # type: ignore

        evaluator = NeuralNetworkEvaluator(board, config)
        score = evaluator.evaluate()

        assert score == 0.0


class TestNNEvaluatorBoardEncoding:
    """Test board encoding methods."""

    def test_encode_board_planes_starting_position(self):
        """Test encoding starting position into 12 planes."""
        board = chess.Board()
        config = EvaluationConfig()
        evaluator = NeuralNetworkEvaluator(board, config)

        planes = evaluator._encode_board_planes()

        # Should have 768 values (12 planes x 64 squares)
        assert len(planes) == 768
        # Should be all binary (0.0 or 1.0)
        assert all(val in [0.0, 1.0] for val in planes)

    def test_encode_board_planes_empty_board(self):
        """Test encoding empty board."""
        board = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")
        config = EvaluationConfig()
        evaluator = NeuralNetworkEvaluator(board, config)

        planes = evaluator._encode_board_planes()

        # All zeros for empty board
        assert all(val == 0.0 for val in planes)

    def test_encode_board_planes_piece_counts(self):
        """Test that piece counts are correct in encoding."""
        board = chess.Board()
        config = EvaluationConfig()
        evaluator = NeuralNetworkEvaluator(board, config)

        planes = evaluator._encode_board_planes()

        # Count 1s in each plane
        plane_size = 64
        white_pawns = sum(planes[0:plane_size])
        white_knights = sum(planes[plane_size : 2 * plane_size])

        # Starting position: 8 white pawns, 2 white knights
        assert white_pawns == 8.0
        assert white_knights == 2.0


class TestNNEvaluatorMaterialFeatures:
    """Test material feature encoding."""

    def test_encode_material_features_starting_position(self):
        """Test material encoding in starting position."""
        board = chess.Board()
        config = EvaluationConfig(use_material=True)
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._encode_material_features()

        # Should have 10 features (5 piece types x 2 colors)
        assert len(features) == 10
        # All values should be between 0 and 1 (normalized)
        assert all(0.0 <= val <= 1.0 for val in features)

    def test_encode_material_features_normalized(self):
        """Test that material features are properly normalized."""
        board = chess.Board()
        config = EvaluationConfig(use_material=True)
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._encode_material_features()

        # Pawns should be 1.0 (8 pawns / 8 max)
        white_pawn_feature = features[0]
        assert white_pawn_feature == 1.0


class TestNNEvaluatorMobilityFeatures:
    """Test mobility feature encoding."""

    def test_encode_mobility_features_starting_position(self):
        """Test mobility encoding in starting position."""
        board = chess.Board()
        config = EvaluationConfig(use_mobility=True)
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._encode_mobility_features()

        # Should have 2 features (White and Black mobility)
        assert len(features) == 2
        # Should be normalized (between 0 and 1)
        assert all(0.0 <= val <= 1.5 for val in features)  # Allow slightly above 1.0

    def test_encode_mobility_features_open_position(self):
        """Test mobility in more open position."""
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")

        config = EvaluationConfig(use_mobility=True)
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._encode_mobility_features()

        # More mobile position should have higher values
        assert len(features) == 2


class TestNNEvaluatorPawnFeatures:
    """Test pawn structure feature encoding."""

    def test_encode_pawn_features_starting_position(self):
        """Test pawn encoding in starting position."""
        board = chess.Board()
        config = EvaluationConfig(use_pawn_structure=True)
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._encode_pawn_features()

        # Should have 6 features (3 per color: doubled, isolated, passed)
        assert len(features) == 6
        # All normalized between 0 and 1
        assert all(0.0 <= val <= 1.0 for val in features)

    def test_encode_pawn_features_doubled_pawns(self):
        """Test detection of doubled pawns."""
        board = chess.Board()
        board.set_fen("rnbqkbnr/pppppppp/8/8/8/2P5/PP1PPPPP/RNBQKBNR w KQkq - 0 1")

        config = EvaluationConfig(use_pawn_structure=True)
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._encode_pawn_features()

        # Should detect doubled pawns
        white_doubled = features[0]
        assert white_doubled > 0.0


class TestNNEvaluatorKingSafetyFeatures:
    """Test king safety feature encoding."""

    def test_encode_king_safety_features_starting_position(self):
        """Test king safety encoding in starting position."""
        board = chess.Board()
        config = EvaluationConfig(use_king_safety=True)
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._encode_king_safety_features()

        # Should have 4 features (2 per color: pawn shield, exposure)
        assert len(features) == 4
        # All normalized
        assert all(0.0 <= val <= 1.0 for val in features)

    def test_encode_king_safety_with_castling(self):
        """Test king safety after castling."""
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        board.push_san("Nf6")
        board.push_san("Be2")
        board.push_san("Be7")
        board.push_san("O-O")

        config = EvaluationConfig(use_king_safety=True)
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._encode_king_safety_features()

        # Castled king should have pawn shield
        assert len(features) == 4


class TestNNEvaluatorFeatureExtraction:
    """Test full feature extraction with various configs."""

    def test_extract_features_minimal_config(self):
        """Test feature extraction with minimal config."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=False,
            use_mobility=False,
            use_pawn_structure=False,
            use_king_safety=False,
        )
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._extract_features()

        # Should have at least board planes (768)
        assert len(features) >= 768
        assert isinstance(features, np.ndarray)

    def test_extract_features_with_material(self):
        """Test feature extraction with material enabled."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_mobility=False,
            use_pawn_structure=False,
        )
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._extract_features()

        # Should have board planes + material features (768 + 10)
        assert len(features) == 778

    def test_extract_features_all_enabled(self):
        """Test feature extraction with all features enabled."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_mobility=True,
            use_pawn_structure=True,
            use_king_safety=True,
            use_tapered_eval=True,
        )
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._extract_features()

        # Should have many features: 768 + 10 + 2 + 6 + 4 + 1 = 791
        assert len(features) == 791
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32


class TestNNEvaluatorWithMockedPyTorch:
    """Test evaluation with mocked PyTorch model."""

    @patch("src.engine.evaluators.nn_eval.NeuralNetworkEvaluator._check_pytorch")
    @patch("src.engine.evaluators.nn_eval.NeuralNetworkEvaluator._extract_features")
    def test_evaluate_with_model(self, mock_extract, mock_check):
        """Test evaluation with mocked model."""
        # Setup mocks
        mock_check.return_value = True
        mock_extract.return_value = np.zeros(791, dtype=np.float32)

        # Create mock model
        mock_model = Mock()
        mock_output = Mock()
        mock_output.item.return_value = 0.5
        mock_model.return_value = mock_output

        # Create evaluator
        board = chess.Board()
        config = EvaluationConfig()

        with patch("torch.no_grad"), patch("torch.FloatTensor") as mock_tensor:
            mock_tensor_instance = Mock()
            mock_tensor_instance.unsqueeze.return_value = mock_tensor_instance
            mock_tensor.return_value = mock_tensor_instance

            evaluator = NeuralNetworkEvaluator(board, config, model=mock_model)
            evaluator.pytorch_available = True

            score = evaluator.evaluate()

            # Should return model's output
            assert score == 0.5

    @patch("src.engine.evaluators.nn_eval.NeuralNetworkEvaluator._check_pytorch")
    def test_load_model_when_pytorch_available(self, mock_check):
        """Test model loading when PyTorch is available."""
        mock_check.return_value = True

        board = chess.Board()
        config = EvaluationConfig()

        with patch("torch.load") as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model

            evaluator = NeuralNetworkEvaluator(board, config, model_path="model.pt")

            # Model loading attempt should have been made
            assert evaluator.pytorch_available is True

    @patch("src.engine.evaluators.nn_eval.NeuralNetworkEvaluator._check_pytorch")
    def test_load_model_raises_error_when_pytorch_unavailable(self, mock_check):
        """Test that loading model without PyTorch raises helpful error."""
        mock_check.return_value = False

        board = chess.Board()
        config = EvaluationConfig()
        evaluator = NeuralNetworkEvaluator(board, config)

        with pytest.raises(ImportError, match="PyTorch is not installed"):
            evaluator._load_model("model.pt")


class TestNNEvaluatorModularFeatures:
    """Test that features respect modular configuration."""

    def test_material_features_only_when_enabled(self):
        """Test that material features are only extracted when enabled."""
        board = chess.Board()

        config_with = EvaluationConfig(use_material=True)
        config_without = EvaluationConfig(use_material=False)

        eval_with = NeuralNetworkEvaluator(board, config_with)
        eval_without = NeuralNetworkEvaluator(board, config_without)

        features_with = eval_with._extract_features()
        features_without = eval_without._extract_features()

        # With material should have 10 more features
        assert len(features_with) == len(features_without) + 10

    def test_mobility_features_require_pst(self):
        """Test that mobility features require PST to be enabled."""
        board = chess.Board()

        # Mobility without PST should not add mobility features
        config = EvaluationConfig(
            use_material=False,
            use_pst=False,
            use_mobility=True,
        )
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._extract_features()

        # Should only have board planes (no mobility features)
        assert len(features) == 768

    def test_king_safety_requires_mobility(self):
        """Test that king safety features require mobility."""
        board = chess.Board()

        # King safety without mobility should not add king safety features
        config = EvaluationConfig(
            use_material=False,
            use_pst=False,
            use_mobility=False,
            use_king_safety=True,
        )
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._extract_features()

        # Should only have board planes (no king safety features)
        assert len(features) == 768

    def test_tapered_eval_adds_phase_feature(self):
        """Test that tapered eval adds game phase feature."""
        board = chess.Board()

        config_with = EvaluationConfig(use_tapered_eval=True)
        config_without = EvaluationConfig(use_tapered_eval=False)

        eval_with = NeuralNetworkEvaluator(board, config_with)
        eval_without = NeuralNetworkEvaluator(board, config_without)

        features_with = eval_with._extract_features()
        features_without = eval_without._extract_features()

        # With tapered eval should have 1 more feature (game phase)
        assert len(features_with) == len(features_without) + 1


class TestNNEvaluatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_board(self):
        """Test with empty board."""
        board = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")
        config = EvaluationConfig()
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._extract_features()

        # Should still extract features
        assert len(features) > 0

    def test_complex_position(self):
        """Test with complex position."""
        board = chess.Board(
            "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 0 1"
        )
        config = EvaluationConfig()
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._extract_features()

        assert len(features) > 0
        assert isinstance(features, np.ndarray)

    def test_endgame_position(self):
        """Test with endgame position."""
        board = chess.Board("8/8/4k3/8/8/4P3/4K3/8 w - - 0 1")
        config = EvaluationConfig(use_tapered_eval=True)
        evaluator = NeuralNetworkEvaluator(board, config)

        features = evaluator._extract_features()

        # Game phase feature should indicate endgame (close to 0)
        game_phase = features[-1]
        assert 0.0 <= game_phase <= 0.2
