# mypy: ignore-errors
# pyright: ignore
# pylint: skip-file
# ruff: noqa

from unittest.mock import Mock, MagicMock, patch

import chess
import torch
import pytest
from torch import nn

from src.engine.config import EngineConfig, MinimaxConfig, EvaluationConfig
from src.engine.evaluators.simple_nn_eval import (  # type: ignore[import-untyped]
    HalfKPNet,
    NeuralNetEvaluator,
)
from src.engine.board.halfkp_representation import (  # type: ignore[import-untyped]
    get_halfkp_feature_size,
    board_to_halfkp_tensor,
    get_piece_index,
    halfkp_index,
    orient_square,
    board_to_input_tensor,
)
from src.engine.search.minimax import Minimax


# These should pass before any changes are made.
class TestPassToPass:
    @pytest.fixture
    def board(self) -> chess.Board:
        return chess.Board()

    @pytest.fixture
    def config(self) -> EngineConfig:
        return EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=False,
                use_iddfs=False,
                use_alpha_beta=True,
                use_move_ordering=False,
                use_pvs=False,
                use_tt_aging=False,
                use_lmr=False,
                max_time=None,
            )
        )

    def test_evaluator_usage(self, board: chess.Board, config: EngineConfig) -> None:
        """Test that the neural network evaluator is actually used during search."""
        mock_evaluator = Mock(spec=NeuralNetEvaluator)
        mock_evaluator.evaluate.return_value = 0.5

        engine = Minimax(board, mock_evaluator, config)
        engine.find_top_move(depth=2)

        assert mock_evaluator.evaluate.called

    def test_feature_size(self) -> None:
        """Test that feature size is calculated correctly."""
        assert get_halfkp_feature_size() == 82048


class TestHalfKPNet:
    """Tests for the HalfKP neural network model."""

    @pytest.fixture
    def model(self) -> HalfKPNet:
        return HalfKPNet()

    @pytest.fixture
    def sample_input(self) -> torch.Tensor:
        return torch.randn(1, 82048)

    def test_model_architecture(self, model: HalfKPNet) -> None:
        """Test that the model has correct 82048->64->1 architecture."""
        assert isinstance(model.hidden, nn.Linear)
        assert isinstance(model.output, nn.Linear)
        assert model.hidden.in_features == 82048
        assert model.hidden.out_features == 64
        assert model.output.in_features == 64
        assert model.output.out_features == 1

    def test_forward_pass_output(
        self, model: HalfKPNet, sample_input: torch.Tensor
    ) -> None:
        """Test that forward pass returns a single float value."""
        output = model(sample_input)
        assert output.shape == (1,)
        assert isinstance(output.item(), float)

    def test_input_shape_validation(self, model: HalfKPNet) -> None:
        """Test that model rejects wrong input shape."""
        wrong_input = torch.randn(1, 1000)
        with pytest.raises(RuntimeError):
            model(wrong_input)

    def test_model_training(self, model: HalfKPNet) -> None:
        """Test that the neural network can actually train and update its weights."""

        # Override NotImplementedError methods for testing
        def initialize_weights(self):
            nn.init.xavier_uniform_(self.hidden.weight)
            nn.init.zeros_(self.hidden.bias)
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward_pass(self, x):
            x = self.hidden(x)
            x = torch.nn.functional.relu(x)  # Using full namespace for consistency
            x = self.output(x)
            return x.squeeze()

        # Monkey patch the methods for testing
        model._initialize_weights = initialize_weights.__get__(model)
        model.forward = forward_pass.__get__(model)
        model._initialize_weights()

        # Create a simple optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Save initial weights for comparison
        initial_hidden_weight = model.hidden.weight.clone().detach()
        initial_output_weight = model.output.weight.clone().detach()

        # Create dummy training data (batch_size=5)
        input_size = 82048  # HalfKP feature size
        inputs = torch.randn(5, input_size)
        targets = torch.tensor([0.5, -0.3, 0.7, -0.2, 0.1])

        # Mini training loop (3 iterations)
        for _ in range(3):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()

        # Check that weights have changed
        assert not torch.allclose(initial_hidden_weight, model.hidden.weight)
        assert not torch.allclose(initial_output_weight, model.output.weight)


class TestNeuralNetEvaluator:
    """Tests for the neural network chess position evaluator."""

    @pytest.fixture
    def board(self) -> chess.Board:
        return chess.Board()

    @pytest.fixture
    def evaluator(self, board: chess.Board) -> NeuralNetEvaluator:
        return NeuralNetEvaluator(board)

    def test_initialization(self, evaluator: NeuralNetEvaluator) -> None:
        """Test that evaluator initializes correctly."""
        assert isinstance(evaluator.model, HalfKPNet)
        assert not evaluator.model.training
        assert evaluator.score is None

    def test_position_sensitivity(self, evaluator: NeuralNetEvaluator) -> None:
        """Test that different positions produce different evaluations."""
        initial_score = evaluator.evaluate()

        evaluator.board.push(chess.Move.from_uci("e2e4"))
        new_score = evaluator.evaluate()

        assert initial_score != new_score

    def test_board_updates(self, evaluator: NeuralNetEvaluator) -> None:
        """Test that board updates work correctly."""
        new_board = chess.Board(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        )
        evaluator.update_board(new_board)
        assert evaluator.board == new_board
        assert evaluator.score is None

    @patch("src.engine.board.halfkp_representation.board_to_input_tensor")
    def test_error_handling(
        self, mock_tensor: MagicMock, evaluator: NeuralNetEvaluator
    ) -> None:
        """Test that evaluator handles tensor conversion errors gracefully."""
        mock_tensor.side_effect = Exception("Tensor conversion failed")
        score = evaluator.evaluate()
        assert isinstance(score, float)

    @patch("src.engine.evaluators.simple_nn_eval.torch.load")
    def test_model_loading_from_path(self, mock_torch_load: MagicMock) -> None:
        """Test that the model attempts to load weights from a given path."""
        mock_board = chess.Board()
        model_path = "/fake/path/to/model.pth"
        mock_weights = {"key": "value"}
        mock_torch_load.return_value = mock_weights

        # Patch the model's load_state_dict method directly on the instance
        with patch.object(HalfKPNet, "load_state_dict") as mock_load_state_dict:
            NeuralNetEvaluator(board=mock_board, model_path=model_path)

            # Assert that torch.load was called with the correct path
            mock_torch_load.assert_called_once_with(model_path, map_location="cpu")

            # Assert that the model's load_state_dict was called with the loaded weights
            mock_load_state_dict.assert_called_once_with(mock_weights)

    def test_no_model_loading_without_path(self) -> None:
        """Test that model loading is not attempted if no path is provided."""
        mock_board = chess.Board()

        with patch(
            "src.engine.evaluators.simple_nn_eval.torch.load"
        ) as mock_torch_load:
            # Instantiate without a model path
            NeuralNetEvaluator(board=mock_board, model_path=None)

            # Assert that torch.load was not called
            mock_torch_load.assert_not_called()


class TestHalfKPRepresentation:
    """Tests for HalfKP chess position representation."""

    def test_square_orientation(self) -> None:
        """Test basic square orientation functionality."""
        assert orient_square(True, 0) == 0
        assert orient_square(False, 0) == 63

    def test_piece_indexing(self) -> None:
        """Test basic piece index calculation."""
        white_pawn = chess.Piece(chess.PAWN, chess.WHITE)
        black_pawn = chess.Piece(chess.PAWN, chess.BLACK)

        assert get_piece_index(white_pawn, True) == 0
        assert get_piece_index(black_pawn, True) == 5

    def test_halfkp_index_range(self) -> None:
        """Test that HalfKP indices are in valid range."""
        piece = chess.Piece(chess.PAWN, chess.WHITE)
        index = halfkp_index(True, 0, 8, piece)
        assert isinstance(index, int)
        assert 0 <= index < 41024

    def test_tensor_conversion_shape(self) -> None:
        """Test that board conversion produces correct tensor shape."""
        board = chess.Board()
        tensor = board_to_input_tensor(board)
        assert tensor.shape == (82048,)
        assert tensor.dtype == torch.float32

    def test_tensor_sparsity(self) -> None:
        """Test that HalfKP tensors are properly sparse."""
        board = chess.Board()
        white_tensor, black_tensor = board_to_halfkp_tensor(board)

        assert torch.sum(white_tensor).item() == 30
        assert torch.sum(black_tensor).item() == 30

    def test_position_differentiation(self) -> None:
        """Test that different board positions produce different tensors."""
        board1 = chess.Board()
        board2 = chess.Board()
        board2.push(chess.Move.from_uci("e2e4"))

        tensor1 = board_to_input_tensor(board1)
        tensor2 = board_to_input_tensor(board2)

        assert not torch.equal(tensor1, tensor2)


class TestMinimaxNeuralNetIntegration:
    """Tests for Minimax engine integration with neural network evaluator."""

    @pytest.fixture
    def board(self) -> chess.Board:
        return chess.Board()

    @pytest.fixture
    def evaluator(self, board: chess.Board) -> NeuralNetEvaluator:
        return NeuralNetEvaluator(board)

    @pytest.fixture
    def config(self) -> EngineConfig:
        return EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=False,
                use_iddfs=False,
                use_alpha_beta=True,
                use_move_ordering=False,
                use_pvs=False,
                use_tt_aging=False,
                use_lmr=False,
                max_time=None,
            )
        )

    def test_basic_integration(
        self, board: chess.Board, evaluator: NeuralNetEvaluator, config: EngineConfig
    ) -> None:
        """Test that Minimax can use NeuralNetEvaluator and find a move."""
        engine = Minimax(board, evaluator, config)  # type: ignore
        score, move = engine.find_top_move(depth=2)

        assert isinstance(score, (int, float))
        assert move is not None
        assert move in board.legal_moves

    def test_board_state_preservation(
        self, board: chess.Board, evaluator: NeuralNetEvaluator, config: EngineConfig
    ) -> None:
        """Test that board state is not modified after search."""
        original_fen = board.fen()

        engine = Minimax(board, evaluator, config)  # type: ignore
        engine.find_top_move(depth=2)

        assert board.fen() == original_fen

    def test_complex_position_search(self, config: EngineConfig) -> None:
        """Test search works from a complex middlegame position."""
        board = chess.Board(
            "r3k2r/ppp2ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/R3K2R w KQkq - 0 1"
        )
        evaluator = NeuralNetEvaluator(board)
        engine = Minimax(board, evaluator, config)  # type: ignore

        score, move = engine.find_top_move(depth=2)

        assert isinstance(score, (int, float))
        assert move is not None
        assert move in board.legal_moves

    def test_terminal_position_handling(self, config: EngineConfig) -> None:
        """Test that terminal positions are handled correctly."""

        board = chess.Board(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        )
        evaluator = NeuralNetEvaluator(board)
        engine = Minimax(board, evaluator, config)  # type: ignore

        score, move = engine.find_top_move(depth=1)

        assert isinstance(score, (int, float))
        assert move is None

    @patch("src.engine.board.halfkp_representation.board_to_input_tensor")
    def test_fallback_evaluation(self, mock_tensor, config: EngineConfig) -> None:
        """Test that fallback evaluation works when tensor conversion fails."""
        mock_tensor.side_effect = Exception("Tensor conversion failed")

        board = chess.Board()
        evaluator = NeuralNetEvaluator(board)
        engine = Minimax(board, evaluator, config)  # type: ignore

        score, move = engine.find_top_move(depth=1)

        assert isinstance(score, (int, float))
        assert move is not None

    def test_search_consistency(
        self, board: chess.Board, evaluator: NeuralNetEvaluator, config: EngineConfig
    ) -> None:
        """Test that multiple searches give consistent results."""
        engine = Minimax(board, evaluator, config)  # type: ignore

        result1 = engine.find_top_move(depth=2)
        result2 = engine.find_top_move(depth=2)

        assert result1 == result2


class TestNeuralNetConfigValidation:
    """Tests for neural network evaluator configuration validation."""

    @pytest.fixture
    def check_nn_available(self):
        """Check if 'nn' is in the allowed evaluator types."""
        from src.engine.config import EvaluationConfig
        import typing

        # Get the type annotation for evaluator_type
        type_hint = typing.get_type_hints(EvaluationConfig)["evaluator_type"]

        # Extract allowed values from the Literal type
        allowed_values = getattr(type_hint, "__args__", [])

        if "nn" not in allowed_values:
            pytest.skip("Neural network evaluator ('nn') is not available in config.py")

    def test_valid_nn_config(self, check_nn_available) -> None:
        """Test that NN evaluator works when all evaluation flags are disabled."""
        config = EngineConfig(
            evaluation=EvaluationConfig(
                evaluator_type="nn",  # type: ignore
                use_material=False,
                use_pst=False,
                use_mobility=False,
                use_pawn_structure=False,
                use_king_safety=False,
            )
        )
        assert config.evaluation.evaluator_type == "nn"

    def test_rejects_single_flag(self, check_nn_available) -> None:
        """Test that NN evaluator rejects individual evaluation flags."""
        with pytest.raises(ValueError):
            EngineConfig(
                evaluation=EvaluationConfig(
                    evaluator_type="nn",  # type: ignore
                    use_material=True,
                    use_pst=False,
                    use_mobility=False,
                    use_pawn_structure=False,
                    use_king_safety=False,
                )
            )

    def test_rejects_multiple_flags(self, check_nn_available) -> None:
        """Test that NN evaluator rejects multiple evaluation flags."""
        with pytest.raises(ValueError):
            EngineConfig(
                evaluation=EvaluationConfig(
                    evaluator_type="nn",  # type: ignore
                    use_material=True,
                    use_pst=True,
                    use_mobility=True,
                    use_pawn_structure=False,
                    use_king_safety=False,
                )
            )

    def test_rejects_default_config(self, check_nn_available) -> None:
        """Test that NN evaluator fails with default config."""
        with pytest.raises(ValueError):
            EngineConfig(evaluation=EvaluationConfig(evaluator_type="nn"))  # type: ignore
