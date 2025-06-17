"""
Test suite for the ComplexEval chess position evaluator.

This module contains tests for the ComplexEval class, which provides
advanced evaluation considering multiple chess factors. Tests cover:
- Individual evaluation components (material, PST, mobility, etc.)
- Game phase transitions and interpolation
- Configuration-driven feature toggling
- Special game states and edge cases
"""

import chess
import pytest
from src.engine.config import EngineConfig, EvaluationConfig
from src.engine.constants import PIECE_VALUES
from src.engine.evaluators.complex_eval import ComplexEval


class TestComplexEvalBasicFunctionality:
    """Test basic functionality of ComplexEval."""

    def test_starting_position_near_zero(self):
        """Test that starting position evaluates close to zero."""
        board = chess.Board()
        config = EvaluationConfig()
        evaluator = ComplexEval(board, config)
        score = evaluator.evaluate()
        # Starting position should be close to 0, but PST may give slight advantage

        assert abs(score) < 1.0

    def test_checkmate_detection(self):
        """Test that checkmate positions return extreme scores."""
        board = chess.Board()
        moves = ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]
        for move_san in moves:
            board.push_san(move_san)
        config = EvaluationConfig()
        evaluator = ComplexEval(board, config)
        score = evaluator.evaluate()

        assert board.is_checkmate()
        assert board.turn == chess.BLACK  # Black is checkmated
        assert score == 9999  # White wins

    def test_stalemate_returns_zero(self):
        """Test that stalemate positions return zero."""
        fen_stalemate = "7k/8/8/8/8/8/2q5/K7 w - - 0 1"
        board = chess.Board(fen_stalemate)
        config = EvaluationConfig()
        evaluator = ComplexEval(board, config)
        score = evaluator.evaluate()
        assert score == 0.0


class TestComplexEvalComponents:
    """Test individual evaluation components."""

    def test_material_only_evaluation(self):
        """Test material evaluation in isolation."""
        board = chess.Board("k7/8/8/8/8/8/R7/K7 w - - 0 1")
        config = EvaluationConfig(
            use_material=True,
            use_pst=False,
            use_mobility=False,
            use_pawn_structure=False,
            use_king_safety=False,
        )
        evaluator = ComplexEval(board, config)
        score = evaluator.evaluate()
        assert score == PIECE_VALUES[chess.ROOK]

    def test_pst_affects_evaluation(self):
        """Test that piece-square tables affect evaluation."""
        board = chess.Board(
            "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"
        )  # Starting position minus knights

        # Place a knight on different squares to see PST effect

        board.remove_piece_at(chess.B1)  # Remove white knight from b1
        board.set_piece_at(
            chess.F3, chess.Piece(chess.KNIGHT, chess.WHITE)
        )  # Place on f3 (better square)

        # With PST

        config_with_pst = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_mobility=False,
            use_pawn_structure=False,
            use_king_safety=False,
        )
        evaluator_with_pst = ComplexEval(board, config_with_pst)
        score_with_pst = evaluator_with_pst.evaluate()

        # Without PST

        config_without_pst = EvaluationConfig(
            use_material=True,
            use_pst=False,
            use_mobility=False,
            use_pawn_structure=False,
            use_king_safety=False,
        )
        evaluator_without_pst = ComplexEval(board, config_without_pst)
        score_without_pst = evaluator_without_pst.evaluate()

        # Scores should be different when PST is enabled

        assert score_with_pst != score_without_pst

    def test_mobility_evaluation(self):
        """Test that mobility affects evaluation."""
        # Position where white has more mobility

        board = chess.Board("k7/pppppppp/8/8/8/8/8/K1Q5 w - - 0 1")

        config = EvaluationConfig(
            use_material=False,
            use_pst=False,
            use_mobility=True,
            use_pawn_structure=False,
            use_king_safety=False,
        )
        evaluator = ComplexEval(board, config)
        score = evaluator.evaluate()

        # White should have mobility advantage

        assert score > 0

    def test_pawn_structure_doubled_pawns(self):
        """Test that doubled pawns are penalized."""
        # White has doubled pawns on f-file

        board = chess.Board("k7/8/8/8/8/5P2/5P2/K7 w - - 0 1")

        config = EvaluationConfig(
            use_material=True,
            use_pst=False,
            use_mobility=False,
            use_pawn_structure=True,
            use_king_safety=False,
        )
        evaluator = ComplexEval(board, config)
        score = evaluator.evaluate()

        # Should be less than just material value due to doubled pawn penalty

        expected_material = 2 * PIECE_VALUES[chess.PAWN]
        assert score < expected_material

    def test_king_safety_pawn_shield(self):
        """Test that pawn shield affects king safety."""
        # Add pieces (e.g., a queen for each side) to create a middlegame phase

        board_with_shield = chess.Board("k1q5/8/8/8/8/8/PPP5/K1R5 w - - 0 1")
        board_exposed = chess.Board("k1q5/8/8/8/8/8/8/K1R5 w - - 0 1")

        config = EvaluationConfig(
            use_material=True,
            use_pst=False,
            use_mobility=False,
            use_pawn_structure=False,
            use_king_safety=True,
        )

        eval_with_shield = ComplexEval(board_with_shield, config)
        eval_exposed = ComplexEval(board_exposed, config)

        score_with_shield = eval_with_shield.evaluate()
        score_exposed = eval_exposed.evaluate()

        # Now, with a non-zero game phase, this assertion should pass

        assert score_with_shield > score_exposed


class TestComplexEvalGamePhase:
    """Test game phase calculation and transitions."""

    def test_opening_game_phase(self):
        """Test that opening position has high game phase value."""
        board = chess.Board()
        config = EvaluationConfig()
        evaluator = ComplexEval(board, config)
        game_phase = evaluator._get_game_phase()

        assert 0.58 <= game_phase <= 0.59

    def test_endgame_phase(self):
        """Test that endgame position has low game phase value."""
        # King and pawn endgame

        board = chess.Board("k7/p7/8/8/8/8/P7/K7 w - - 0 1")
        config = EvaluationConfig()
        evaluator = ComplexEval(board, config)
        game_phase = evaluator._get_game_phase()

        # Endgame should have very low game phase

        assert game_phase < 0.2

    def test_middlegame_phase(self):
        """Test middlegame position has intermediate game phase."""
        # Remove some pieces to simulate middlegame

        board = chess.Board(
            "r3k2r/ppp2ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/R3K2R w KQkq - 0 1"
        )
        config = EvaluationConfig()
        evaluator = ComplexEval(board, config)
        game_phase = evaluator._get_game_phase()

        # Middlegame should be between opening and endgame

        assert 0.3 < game_phase < 0.8


class TestComplexEvalConfiguration:
    """Test configuration-driven feature toggling."""

    def test_all_features_disabled_gives_zero(self):
        """Test that disabling all features gives zero evaluation."""
        board = chess.Board("k1q5/8/8/8/8/8/R7/K7 w - - 0 1")  # Material imbalance

        config = EvaluationConfig(
            use_material=False,
            use_pst=False,
            use_mobility=False,
            use_pawn_structure=False,
            use_king_safety=False,
        )
        evaluator = ComplexEval(board, config)
        score = evaluator.evaluate()
        assert score == 0.0

    def test_feature_combinations(self):
        """Test that different feature combinations give different evaluations."""
        board = chess.Board(
            "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
        )

        # Material + PST

        config1 = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_mobility=False,
            use_pawn_structure=False,
            use_king_safety=False,
        )

        # Material + Mobility

        config2 = EvaluationConfig(
            use_material=True,
            use_pst=False,
            use_mobility=True,
            use_pawn_structure=False,
            use_king_safety=False,
        )

        eval1 = ComplexEval(board, config1)
        eval2 = ComplexEval(board, config2)

        score1 = eval1.evaluate()
        score2 = eval2.evaluate()

        # Different configurations should give different scores

        assert score1 != score2

    def test_config_validation_complex_evaluator(self):
        """Test that complex evaluator requires at least one feature."""
        with pytest.raises(
            ValueError,
            match="Complex evaluator must have at least one evaluation feature enabled",
        ):
            EngineConfig(
                evaluation=EvaluationConfig(
                    evaluator_type="complex",
                    use_material=False,
                    use_pst=False,
                    use_mobility=False,
                    use_pawn_structure=False,
                    use_king_safety=False,
                )
            )


class TestComplexEvalEdgeCases:
    """Test edge cases and special scenarios."""

    def test_insufficient_material_draw(self):
        """Test that insufficient material returns zero."""
        # King vs King + Bishop (insufficient material)

        board = chess.Board("8/8/8/8/8/k7/b7/K7 w - - 0 1")
        config = EvaluationConfig()
        evaluator = ComplexEval(board, config)
        score = evaluator.evaluate()
        assert score == 0.0

    def test_extreme_material_imbalance(self):
        """Test evaluation with extreme material differences."""
        # White has queen, black has only pawns

        board = chess.Board("k7/pppppppp/8/8/8/8/8/K1Q5 w - - 0 1")
        config = EvaluationConfig(
            use_material=True,
            use_pst=False,
            use_mobility=False,
            use_pawn_structure=False,
            use_king_safety=False,
        )
        evaluator = ComplexEval(board, config)
        score = evaluator.evaluate()

        expected_diff = PIECE_VALUES[chess.QUEEN] - 8 * PIECE_VALUES[chess.PAWN]
        assert abs(score - expected_diff) < 0.1

    def test_turn_preservation(self):
        """Test that evaluation doesn't permanently change board turn."""
        board = chess.Board()
        original_turn = board.turn

        config = EvaluationConfig(use_mobility=True)
        evaluator = ComplexEval(board, config)
        evaluator.evaluate()

        # Turn should be restored after evaluation

        assert board.turn == original_turn

    def test_pawn_file_calculation(self):
        """Test that pawn file structure is calculated correctly."""
        # Specific pawn structure

        board = chess.Board("k7/p1p1p3/8/8/8/8/P1P1P3/K7 w - - 0 1")
        config = EvaluationConfig()
        evaluator = ComplexEval(board, config)

        # Check that pawn files are calculated correctly

        assert evaluator.pawn_files[chess.WHITE][0] == 1  # a-file
        assert evaluator.pawn_files[chess.WHITE][1] == 0  # b-file
        assert evaluator.pawn_files[chess.WHITE][2] == 1  # c-file
        assert evaluator.pawn_files[chess.WHITE][4] == 1  # e-file


class TestComplexEvalIntegration:
    """Test integration aspects of ComplexEval."""

    def test_consistent_evaluation(self):
        """Test that evaluation is consistent across multiple calls."""
        board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
        )
        config = EvaluationConfig()
        evaluator = ComplexEval(board, config)

        # Multiple evaluations should give same result

        scores = [evaluator.evaluate() for _ in range(3)]
        assert all(score == scores[0] for score in scores)

    def test_config_object_independence(self):
        """Test that different config objects work independently."""
        board = chess.Board()

        config1 = EvaluationConfig(use_material=True, use_pst=False)
        config2 = EvaluationConfig(use_material=False, use_pst=True)

        eval1 = ComplexEval(board, config1)
        eval2 = ComplexEval(board, config2)

        # Should not interfere with each other

        score1 = eval1.evaluate()
        score2 = eval2.evaluate()

        # Verify configs are still different

        assert config1.use_material != config2.use_material
        assert config1.use_pst != config2.use_pst


if __name__ == "__main__":
    pytest.main([__file__])
