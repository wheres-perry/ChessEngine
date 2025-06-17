"""
Test suite for the SimpleEval chess position evaluator.

This module contains comprehensive tests for the SimpleEval class, which provides
material-based evaluation of chess positions. Tests cover:
- Basic material counting
- Special game states (checkmate, stalemate, draws)
- Edge cases and error conditions
- Configuration validation
"""

import chess
import pytest
from src.engine.config import EngineConfig, EvaluationConfig
from src.engine.constants import PIECE_VALUES
from src.engine.evaluators.simple_eval import SimpleEval


class TestSimpleEvalBasicFunctionality:
    """Test basic evaluation functionality of SimpleEval."""

    def test_starting_position_equal_material(self):
        """Test that the starting position evaluates to zero (equal material)."""
        board = chess.Board()
        evaluator = SimpleEval(board)
        score = evaluator.evaluate()
        assert score == 0.0

    def test_kings_only_position(self):
        """Test that a position with only kings evaluates to zero."""
        board = chess.Board("k7/8/K7/8/8/8/8/8 w - - 0 1")
        evaluator = SimpleEval(board)
        score = evaluator.evaluate()
        assert score == 0.0

    def test_material_advantage_white_single_piece(self):
        """Test white material advantage with a single extra piece."""
        board = chess.Board("k7/8/8/8/8/8/R7/K7 w - - 0 1")
        evaluator = SimpleEval(board)
        score = evaluator.evaluate()
        expected_score = PIECE_VALUES[chess.ROOK]
        assert score == expected_score

    def test_material_advantage_black_single_piece(self):
        """Test black material advantage with a single extra piece."""
        board = chess.Board("k1q5/8/8/8/8/8/8/K7 w - - 0 1")
        evaluator = SimpleEval(board)
        score = evaluator.evaluate()
        expected_score = -PIECE_VALUES[chess.QUEEN]
        assert score == expected_score

    def test_complex_material_advantage_white(self):
        """Test white material advantage with multiple pieces."""
        board = chess.Board("8/8/8/8/8/k7/B7/K1Q5 w - - 0 1")
        evaluator = SimpleEval(board)
        expected_score = PIECE_VALUES[chess.QUEEN] + PIECE_VALUES[chess.BISHOP]
        score = evaluator.evaluate()
        assert score == expected_score

    def test_mixed_material_difference(self):
        """Test position where both sides have different pieces."""
        # White has Queen, Black has Rook and Bishop

        board = chess.Board("k1rb4/8/8/8/8/8/8/K1Q5 w - - 0 1")
        evaluator = SimpleEval(board)
        white_material = PIECE_VALUES[chess.QUEEN]
        black_material = PIECE_VALUES[chess.ROOK] + PIECE_VALUES[chess.BISHOP]
        expected_score = white_material - black_material
        score = evaluator.evaluate()
        assert score == expected_score


class TestSimpleEvalGameStates:
    """Test evaluation of special game states."""

    def test_white_checkmate_advantage(self):
        """Test that white delivering checkmate gets positive infinity."""
        board = chess.Board()
        moves = ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]
        for move_san in moves:
            board.push_san(move_san)
        evaluator = SimpleEval(board)
        score = evaluator.evaluate()
        assert board.is_checkmate()
        assert board.turn == chess.BLACK  # Black is checkmated
        assert score == float("inf")

    def test_black_checkmate_advantage(self):
        """Test that black delivering checkmate gets negative infinity."""
        board = chess.Board()
        moves = ["f3", "e5", "g4", "Qh4#"]
        for move_san in moves:
            board.push_san(move_san)
        evaluator = SimpleEval(board)
        score = evaluator.evaluate()
        assert board.is_checkmate()
        assert board.turn == chess.WHITE  # White is checkmated
        assert score == float("-inf")

    def test_stalemate_position(self):
        """Test that stalemate positions evaluate to zero."""
        fen_stalemate = "7k/8/8/8/8/8/2q5/K7 w - - 0 1"
        board = chess.Board(fen_stalemate)
        assert board.is_stalemate()
        evaluator = SimpleEval(board)
        score = evaluator.evaluate()
        assert score == 0.0

    def test_insufficient_material_draw(self):
        """Test positions with insufficient material to checkmate."""
        # King vs King + Bishop

        board = chess.Board("8/8/8/8/8/k7/b7/K7 w - - 0 1")
        evaluator = SimpleEval(board)
        score = evaluator.evaluate()
        # Position has insufficient material to checkmate, should return 0

        assert score == 0.0


class TestSimpleEvalPieceTypes:
    """Test evaluation with different piece types."""

    @pytest.mark.parametrize(
        "piece_type,expected_value",
        [
            (chess.PAWN, PIECE_VALUES[chess.PAWN]),
            (chess.KNIGHT, PIECE_VALUES[chess.KNIGHT]),
            (chess.BISHOP, PIECE_VALUES[chess.BISHOP]),
            (chess.ROOK, PIECE_VALUES[chess.ROOK]),
            (chess.QUEEN, PIECE_VALUES[chess.QUEEN]),
        ],
    )
    def test_individual_piece_values(self, piece_type, expected_value):
        """Test that individual piece types are valued correctly."""
        piece_symbol = chess.piece_symbol(piece_type).upper()

        fen = f"k7/8/8/8/8/8/{piece_symbol}7/KP6 w - - 0 1"

        board = chess.Board(fen)
        evaluator = SimpleEval(board)
        score = evaluator.evaluate()

        # Account for the pawn in the expected score

        expected_score = expected_value + PIECE_VALUES[chess.PAWN]
        assert score == expected_score

    def test_multiple_same_pieces(self):
        """Test evaluation with multiple pieces of the same type."""
        # White has 3 pawns, Black has 1 pawn

        board = chess.Board("k7/p7/8/8/8/8/PPP5/K7 w - - 0 1")
        evaluator = SimpleEval(board)
        expected_score = 2 * PIECE_VALUES[chess.PAWN]  # Net difference
        score = evaluator.evaluate()
        assert score == expected_score

    def test_promoted_pieces(self):
        """Test evaluation with promoted pieces."""
        # Pawn promoted to queen

        board = chess.Board("k6Q/8/8/8/8/8/8/K7 w - - 0 1")
        evaluator = SimpleEval(board)
        expected_score = PIECE_VALUES[chess.QUEEN]
        score = evaluator.evaluate()
        assert score == expected_score


class TestSimpleEvalConfiguration:
    """Test configuration and integration aspects of SimpleEval."""

    def test_simple_eval_ignores_complex_flags(self):
        """Test that SimpleEval doesn't break when complex flags are present in config."""
        board = chess.Board()

        # Create config with complex evaluator flags (should be ignored by SimpleEval)

        config = EvaluationConfig(
            evaluator_type="simple",
            use_material=True,
            use_pst=True,
            use_mobility=True,
            use_pawn_structure=True,
            use_king_safety=True,
        )

        # SimpleEval doesn't use config, so this should work fine

        evaluator = SimpleEval(board)
        score = evaluator.evaluate()
        assert score == 0.0

    def test_engine_config_validation_complex_flags_with_simple_eval(self):
        """Test that engine validates when complex flags are used with simple evaluator."""
        # This test assumes there should be validation logic in the engine
        # If not implemented yet, this documents the expected behavior

        config = EngineConfig()
        config.evaluation.evaluator_type = "simple"
        config.evaluation.use_pst = True  # This should be flagged as invalid

        # For now, just verify the configuration is created
        # In the future, this should raise a validation error

        assert config.evaluation.evaluator_type == "simple"
        assert config.evaluation.use_pst == True

        # TODO: Add validation logic to EngineConfig to prevent this invalid combination
        # with pytest.raises(ValueError, match="Complex evaluation flags cannot be used with simple evaluator"):
        #     validate_engine_config(config)


class TestSimpleEvalEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_board_except_kings(self):
        """Test evaluation with only kings on the board."""
        board = chess.Board("8/8/8/3k4/3K4/8/8/8 w - - 0 1")
        evaluator = SimpleEval(board)
        score = evaluator.evaluate()
        assert score == 0.0

    def test_maximum_material_difference(self):
        """Test position with maximum possible material difference."""
        # White has all pieces, Black has only king

        board = chess.Board("k7/8/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
        evaluator = SimpleEval(board)

        expected_score = (
            8 * PIECE_VALUES[chess.PAWN]
            + 2 * PIECE_VALUES[chess.ROOK]
            + 2 * PIECE_VALUES[chess.KNIGHT]
            + 2 * PIECE_VALUES[chess.BISHOP]
            + 1 * PIECE_VALUES[chess.QUEEN]
        )

        score = evaluator.evaluate()
        assert score == expected_score

    def test_evaluator_state_independence(self):
        """Test that evaluator doesn't maintain state between evaluations."""
        board1 = chess.Board("k7/8/8/8/8/8/R7/K7 w - - 0 1")
        board2 = chess.Board("k1q5/8/8/8/8/8/8/K7 w - - 0 1")

        evaluator = SimpleEval(board1)
        score1 = evaluator.evaluate()

        # Change the board and evaluate again

        evaluator.board = board2
        score2 = evaluator.evaluate()

        assert score1 == PIECE_VALUES[chess.ROOK]
        assert score2 == -PIECE_VALUES[chess.QUEEN]

    def test_fen_position_consistency(self):
        """Test that the same position via different FENs gives same evaluation."""
        # Starting position via default constructor

        board1 = chess.Board()

        # Starting position via FEN

        board2 = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

        evaluator1 = SimpleEval(board1)
        evaluator2 = SimpleEval(board2)

        assert evaluator1.evaluate() == evaluator2.evaluate() == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
