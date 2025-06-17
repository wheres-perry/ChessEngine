"""
Test suite for the MockEval chess position evaluator.

This module contains tests for the MockEval class, which provides
a controllable mock evaluator for testing purposes. Tests cover:
- Core functionality and score modification
- Board state independence (key feature)
- Configuration validation
- Edge cases and bug detection
"""

import chess
import pytest
from src.engine.config import EngineConfig, EvaluationConfig
from src.engine.evaluators.mock_eval import MockEval


class TestMockEvalCoreFunctionality:
    """Test core functionality of MockEval."""

    def test_default_score_zero(self):
        """Test that MockEval returns zero by default."""
        board = chess.Board()
        evaluator = MockEval(board)
        score = evaluator.evaluate()
        assert score == 0.0

    def test_custom_initial_score(self):
        """Test that MockEval can be initialized with a custom score."""
        board = chess.Board()
        evaluator = MockEval(board, fixed_score=42.5)
        score = evaluator.evaluate()
        assert score == 42.5

    def test_set_score_method(self):
        """Test that set_score() method updates the evaluation."""
        board = chess.Board()
        evaluator = MockEval(board, fixed_score=10.0)

        assert evaluator.evaluate() == 10.0

        evaluator.set_score(25.0)
        assert evaluator.evaluate() == 25.0

    def test_multiple_score_updates(self):
        """Test multiple consecutive score updates work correctly."""
        board = chess.Board()
        evaluator = MockEval(board)

        test_scores = [15.5, -30.0, 100.0, 0.0]
        for expected_score in test_scores:
            evaluator.set_score(expected_score)
            assert evaluator.evaluate() == expected_score

    def test_infinity_scores(self):
        """Test MockEval with infinite score values (important for checkmate testing)."""
        board = chess.Board()
        evaluator = MockEval(board)

        evaluator.set_score(float("inf"))
        assert evaluator.evaluate() == float("inf")

        evaluator.set_score(float("-inf"))
        assert evaluator.evaluate() == float("-inf")


class TestMockEvalBoardIndependence:
    """Test that MockEval is independent of board state - the key feature."""

    def test_checkmate_position_ignores_board(self):
        """Test that checkmate positions don't affect evaluation."""
        board = chess.Board()
        moves = ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]
        for move_san in moves:
            board.push_san(move_san)
        assert board.is_checkmate()

        evaluator = MockEval(board, fixed_score=123.0)
        score = evaluator.evaluate()
        assert score == 123.0  # Should ignore checkmate

    def test_material_imbalance_ignores_board(self):
        """Test that material imbalances don't affect evaluation."""
        # White has significant material advantage

        board = chess.Board("k7/8/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")

        evaluator = MockEval(board, fixed_score=0.0)
        score = evaluator.evaluate()
        assert score == 0.0  # Should ignore material advantage

    def test_board_modification_doesnt_affect_score(self):
        """Test that modifying the board doesn't change the evaluation."""
        board = chess.Board()
        evaluator = MockEval(board, fixed_score=50.0)

        assert evaluator.evaluate() == 50.0

        # Make moves on the board

        board.push_san("e4")
        board.push_san("e5")

        # Evaluation should remain the same

        assert evaluator.evaluate() == 50.0


class TestMockEvalConfiguration:
    """Test configuration validation for MockEval."""

    def test_mock_eval_with_any_flags_raises_error(self):
        """Test that using any evaluation flags with mock evaluator raises validation error."""
        with pytest.raises(
            ValueError, match="Evaluation flags.*cannot be used with mock evaluator"
        ):
            EngineConfig(
                evaluation=EvaluationConfig(evaluator_type="mock", use_material=True)
            )

    def test_valid_mock_eval_config(self):
        """Test that valid mock evaluator configuration works."""
        config = EngineConfig(
            evaluation=EvaluationConfig(
                evaluator_type="mock",
                use_material=False,
                use_pst=False,
                use_mobility=False,
                use_pawn_structure=False,
                use_king_safety=False,
            )
        )
        assert config.evaluation.evaluator_type == "mock"

    def test_mock_eval_default_config_is_invalid(self):
        """Test that default config with mock evaluator type raises error due to enabled flags."""
        with pytest.raises(
            ValueError, match="Evaluation flags.*cannot be used with mock evaluator"
        ):
            EngineConfig(
                evaluation=EvaluationConfig(evaluator_type="mock")
                # Default config has use_material=True and other flags enabled
            )


class TestMockEvalTestingUtility:
    """Test MockEval's utility for testing scenarios."""

    @pytest.mark.parametrize("test_score", [-1000.0, -0.1, 0.0, 0.1, 1000.0])
    def test_parametrized_scores(self, test_score):
        """Test MockEval with various score values important for testing."""
        board = chess.Board()
        evaluator = MockEval(board, fixed_score=test_score)
        assert evaluator.evaluate() == test_score

    def test_state_isolation_between_instances(self):
        """Test that different MockEval instances don't affect each other."""
        board = chess.Board()

        evaluator1 = MockEval(board, fixed_score=100.0)
        evaluator2 = MockEval(board, fixed_score=200.0)

        evaluator1.set_score(150.0)

        # Other evaluator should be unaffected

        assert evaluator1.evaluate() == 150.0
        assert evaluator2.evaluate() == 200.0


class TestMockEvalBugDetection:
    """Test for bugs and edge cases in MockEval."""

    def test_set_score_updates_score_attribute(self):
        """Test for the bug where set_score doesn't update self.score properly."""
        board = chess.Board()
        evaluator = MockEval(board, fixed_score=10.0)

        evaluator.set_score(20.0)

        # Both the return value and attribute should be updated

        assert evaluator.evaluate() == 20.0
        assert evaluator.score == 20.0

    def test_scientific_notation_scores(self):
        """Test MockEval with extreme values that might be used in testing."""
        board = chess.Board()

        # Very large number (beyond typical chess scores)

        large_score = 1e6
        evaluator = MockEval(board, fixed_score=large_score)
        assert evaluator.evaluate() == large_score

        # Very small number

        small_score = 1e-6
        evaluator.set_score(small_score)
        assert evaluator.evaluate() == small_score


if __name__ == "__main__":
    pytest.main([__file__])
