"""
Comprehensive tests for BaseEvaluator and its concrete implementations.

Tests cover:
- BaseEvaluator utility methods
- SimpleEvaluator functionality
- MockEvaluator functionality
- Configuration validation
- Edge cases and error conditions
"""

import chess as pychess
import pytest

from engine._core import chess_engine_core as chess
from src.engine.config import EvaluationConfig
from src.engine.evaluators.base_evaluator import (
    BaseEvaluator,
    MockEvaluator,
    SimpleEvaluator,
)


class TestBaseEvaluatorUtilities:
    """Test BaseEvaluator utility methods."""

    def test_get_game_phase_opening(self):
        """Test game phase calculation in opening position."""
        board = chess.Board()
        config = EvaluationConfig()
        evaluator = SimpleEvaluator(board, config)

        phase = evaluator.get_game_phase()

        # Opening should be close to 1.0 (all pieces present)
        assert 0.9 <= phase <= 1.0

    def test_get_game_phase_endgame(self):
        """Test game phase calculation in endgame."""
        # King and pawn endgame
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        config = EvaluationConfig()
        evaluator = SimpleEvaluator(board, config)

        phase = evaluator.get_game_phase()

        # Endgame should be close to 0.0
        assert 0.0 <= phase <= 0.2

    def test_get_game_phase_middlegame(self):
        """Test game phase calculation in middlegame."""
        # Middlegame with some pieces traded
        board = chess.Board(
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
        )
        config = EvaluationConfig()
        evaluator = SimpleEvaluator(board, config)

        phase = evaluator.get_game_phase()

        # This position still has most pieces, so phase is high
        assert 0.8 <= phase <= 1.0

    def test_count_material_starting_position(self):
        """Test material counting in starting position."""
        board = chess.Board()
        config = EvaluationConfig()
        evaluator = SimpleEvaluator(board, config)

        white_material, black_material = evaluator.count_material()

        # Starting position: equal material
        assert white_material == black_material
        # Total material: 8 pawns (800) + 2 knights (600) + 2 bishops (600)
        # + 2 rooks (1000) + 1 queen (900) = 4000
        # (Note: Actual values may vary by piece value table)
        assert white_material == 4000
        assert black_material == 4000

    def test_count_material_imbalanced(self):
        """Test material counting with imbalance."""
        # White up a queen
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        config = EvaluationConfig()
        evaluator = SimpleEvaluator(board, config)

        white_material, black_material = evaluator.count_material()

        # White should have 900 more centipawns (queen)
        assert white_material == black_material + 900

    def test_count_material_matches_python_chess(self):
        """Ensure material counting matches python-chess reference implementation."""
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
        board_cpp = chess.Board.from_fen(fen)
        board_py = pychess.Board(fen)
        evaluator = SimpleEvaluator(board_cpp, EvaluationConfig())

        cpp_white, cpp_black = evaluator.count_material()

        piece_values = {
            pychess.PAWN: 100,
            pychess.KNIGHT: 320,
            pychess.BISHOP: 330,
            pychess.ROOK: 500,
            pychess.QUEEN: 900,
        }

        def material_py(color: bool) -> int:
            total = 0
            for piece_type, value in piece_values.items():
                total += len(board_py.pieces(piece_type, color)) * value
            return total

        assert cpp_white == material_py(pychess.WHITE)
        assert cpp_black == material_py(pychess.BLACK)

    def test_get_piece_count(self):
        """Test piece counting."""
        board = chess.Board()
        config = EvaluationConfig()
        evaluator = SimpleEvaluator(board, config)

        # Starting position counts
        assert evaluator.get_piece_count(chess.PAWN, chess.WHITE) == 8
        assert evaluator.get_piece_count(chess.KNIGHT, chess.WHITE) == 2
        assert evaluator.get_piece_count(chess.BISHOP, chess.WHITE) == 2
        assert evaluator.get_piece_count(chess.ROOK, chess.WHITE) == 2
        assert evaluator.get_piece_count(chess.QUEEN, chess.WHITE) == 1
        assert evaluator.get_piece_count(chess.KING, chess.WHITE) == 1

    def test_is_endgame_with_queens(self):
        """Test endgame detection with queens on board."""
        board = chess.Board()
        config = EvaluationConfig()
        evaluator = SimpleEvaluator(board, config)

        # With queens, not endgame
        assert not evaluator.is_endgame()

    def test_is_endgame_without_queens(self):
        """Test endgame detection without queens."""
        # No queens
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
        config = EvaluationConfig()
        evaluator = SimpleEvaluator(board, config)

        # Without queens and low material, should be endgame
        # But this still has a lot of material, so might not be endgame
        # Let's test a clearer endgame
        board_endgame = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")
        evaluator_endgame = SimpleEvaluator(board_endgame, config)

        assert evaluator_endgame.is_endgame()

    def test_get_piece_mobility(self):
        """Test piece mobility calculation."""
        board = chess.Board()
        config = EvaluationConfig()
        evaluator = SimpleEvaluator(board, config)

        # Knight on b1 in starting position has 2 moves
        knight_square = chess.B1
        mobility = evaluator.get_piece_mobility(knight_square)

        assert mobility == 2

    def test_get_piece_mobility_empty_square(self):
        """Test mobility for empty square."""
        board = chess.Board()
        config = EvaluationConfig()
        evaluator = SimpleEvaluator(board, config)

        # Empty square should return 0
        empty_square = chess.E4
        mobility = evaluator.get_piece_mobility(empty_square)

        assert mobility == 0

    def test_square_to_index_white_perspective(self):
        """Test square to index conversion for White."""
        board = chess.Board()
        config = EvaluationConfig()
        evaluator = SimpleEvaluator(board, config)

        # A1 should be index 0 for White
        assert evaluator.square_to_index(chess.A1, flip_for_black=False) == 0
        # H8 should be index 63 for White
        assert evaluator.square_to_index(chess.H8, flip_for_black=False) == 63

    def test_square_to_index_black_perspective(self):
        """Test square to index conversion for Black (flipped)."""
        board = chess.Board()
        config = EvaluationConfig()
        evaluator = SimpleEvaluator(board, config)

        # A1 from Black's perspective is flipped
        flipped = evaluator.square_to_index(chess.A1, flip_for_black=True)
        assert flipped == chess.square_mirror(chess.A1)

    def test_interpolate_with_tapered_eval_enabled(self):
        """Test interpolation with tapered eval enabled."""
        board = chess.Board()
        config = EvaluationConfig(use_tapered_eval=True)
        evaluator = SimpleEvaluator(board, config)

        mg_value = 100.0
        eg_value = 50.0

        interpolated = evaluator.interpolate(mg_value, eg_value)

        # Should be between mg and eg values
        assert eg_value <= interpolated <= mg_value

    def test_interpolate_with_tapered_eval_disabled(self):
        """Test interpolation with tapered eval disabled."""
        board = chess.Board()
        config = EvaluationConfig(use_tapered_eval=False)
        evaluator = SimpleEvaluator(board, config)

        mg_value = 100.0
        eg_value = 50.0

        interpolated = evaluator.interpolate(mg_value, eg_value)

        # Should return mg value
        assert interpolated == mg_value


class TestSimpleEvaluator:
    """Test SimpleEvaluator (material-only evaluation)."""

    def test_evaluate_starting_position(self):
        """Test evaluation of starting position."""
        board = chess.Board()
        evaluator = SimpleEvaluator(board, None)

        score = evaluator.evaluate()

        # Starting position is equal
        assert score == 0.0

    def test_evaluate_white_advantage(self):
        """Test evaluation with White material advantage."""
        # White up a queen
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        evaluator = SimpleEvaluator(board, None)

        score = evaluator.evaluate()

        # White should be up by 9.0 (queen = 900 centipawns = 9 pawns)
        assert score == 9.0

    def test_evaluate_black_advantage(self):
        """Test evaluation with Black material advantage."""
        # Black up a rook
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1")
        evaluator = SimpleEvaluator(board, None)

        score = evaluator.evaluate()

        # Black should be up by 5.0 (rook = 500 centipawns = 5 pawns)
        assert score == -5.0

    def test_config_defaults_to_material_only(self):
        """Test that SimpleEvaluator uses material-only config."""
        board = chess.Board()
        evaluator = SimpleEvaluator(board, None)

        # Should have material enabled, everything else disabled
        assert evaluator.config.use_material is True
        assert evaluator.config.use_pst is False
        assert evaluator.config.use_mobility is False


class TestMockEvaluator:
    """Test MockEvaluator (returns fixed value)."""

    def test_evaluate_returns_zero(self):
        """Test that MockEvaluator returns 0.0."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)

        score = evaluator.evaluate()

        assert score == 0.0

    def test_evaluate_consistent(self):
        """Test that MockEvaluator always returns same value."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)

        # Evaluate multiple times
        scores = [evaluator.evaluate() for _ in range(5)]

        # All should be 0.0
        assert all(score == 0.0 for score in scores)

    def test_evaluate_regardless_of_position(self):
        """Test that MockEvaluator ignores position."""
        board1 = chess.Board()  # Starting position
        board2 = chess.Board("8/8/8/8/8/8/8/k6K w - - 0 1")  # Endgame

        evaluator1 = MockEvaluator(board1, None)
        evaluator2 = MockEvaluator(board2, None)

        # Both should return 0.0
        assert evaluator1.evaluate() == 0.0
        assert evaluator2.evaluate() == 0.0

    def test_config_defaults_to_mock(self):
        """Test that MockEvaluator uses mock config."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)

        assert evaluator.config.evaluator_type == "mock"


class TestBaseEvaluatorConfigValidation:
    """Test configuration validation in BaseEvaluator."""

    def test_validate_config_valid(self):
        """Test that valid config passes validation."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_tapered_eval=True,
        )

        # Should not raise
        evaluator = SimpleEvaluator(board, config)
        assert evaluator is not None

    def test_validate_config_called_on_init(self):
        """Test that validation is called during initialization."""
        board = chess.Board()

        # Valid config
        config_valid = EvaluationConfig(use_material=True, use_pst=False)
        evaluator = SimpleEvaluator(board, config_valid)

        # Validation should have been called (no exception raised)
        assert evaluator.config == config_valid


class TestBaseEvaluatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_board(self):
        """Test evaluation with empty board."""
        board = chess.Board(fen="8/8/8/8/8/8/8/8 w - - 0 1")
        evaluator = SimpleEvaluator(board, None)

        score = evaluator.evaluate()

        # No pieces, no material
        assert score == 0.0

    def test_only_kings(self):
        """Test with only kings on board."""
        board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
        evaluator = SimpleEvaluator(board, None)

        white_material, black_material = evaluator.count_material()

        # Kings have 0 material value
        assert white_material == 0
        assert black_material == 0

    def test_game_phase_with_only_kings(self):
        """Test game phase with only kings."""
        board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
        evaluator = SimpleEvaluator(board, None)

        phase = evaluator.get_game_phase()

        # No pieces means phase should be 0 (endgame)
        assert phase == 0.0

    def test_maximum_material(self):
        """Test with maximum material (promoted pieces)."""
        # Impossible position but valid for testing
        board = chess.Board("QQQQQQQQ/PPPPPPPP/8/8/8/8/pppppppp/qqqqqqqq w - - 0 1")
        evaluator = SimpleEvaluator(board, None)

        white_material, black_material = evaluator.count_material()

        # Should handle large material counts
        assert white_material > 0
        assert black_material > 0
        assert white_material == black_material

    def test_piece_mobility_king(self):
        """Test king mobility in corner."""
        board = chess.Board("K7/8/8/8/8/8/8/8 w - - 0 1")
        evaluator = SimpleEvaluator(board, None)

        mobility = evaluator.get_piece_mobility(chess.A8)

        # King in corner has 3 moves
        assert mobility == 3

    def test_piece_mobility_queen_center(self):
        """Test queen mobility in center of empty board."""
        board = chess.Board("8/8/8/8/3Q4/8/8/K6k w - - 0 1")
        evaluator = SimpleEvaluator(board, None)

        mobility = evaluator.get_piece_mobility(chess.D4)

        # Queen in center should have many moves (26 in this position)
        assert mobility >= 25
