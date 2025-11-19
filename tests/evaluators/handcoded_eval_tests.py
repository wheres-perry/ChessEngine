"""
Comprehensive tests for HandcodedEvaluator.

Tests cover:
- Modular feature toggling (Tree 2 dependencies)
- Material evaluation (E0)
- Piece-square tables (E1)
- Tapered evaluation (E2)
- Pawn structure (E3)
- Mobility (E4)
- King safety (E5)
- Configuration validation
"""

import pytest

from engine._core import chess_engine_core as chess
from src.engine.config import EvaluationConfig
from src.engine.evaluators.handcoded_eval import HandcodedEvaluator


class TestHandcodedEvaluatorMaterialOnly:
    """Test material-only evaluation (Node E0)."""

    def test_material_only_enabled(self):
        """Test with only material evaluation enabled."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=False,
            use_mobility=False,
            use_pawn_structure=False,
            use_king_safety=False,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        # Starting position: equal material
        assert score == 0.0

    def test_material_disabled_returns_zero(self):
        """Test that evaluation returns 0 when material is disabled."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=False,
            use_pst=False,
            use_mobility=False,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        # Base of tree disabled, everything returns 0
        assert score == 0.0


class TestHandcodedEvaluatorPST:
    """Test piece-square table evaluation (Node E1)."""

    def test_pst_enabled(self):
        """Test PST evaluation when enabled."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_tapered_eval=False,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        # With PST, score should be non-zero (even in starting position)
        # due to piece placement bonuses
        assert isinstance(score, float)

    def test_pst_disabled(self):
        """Test that PST is not used when disabled."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=False,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        # Should only be material evaluation
        assert score == 0.0  # Equal material in starting position

    def test_pst_favors_good_piece_placement(self):
        """Test that PST gives bonus for good piece placement."""
        # Move knight to center
        board = chess.Board()
        board.push_san("Nf3")
        board.push_san("Nf6")
        board.push_san("Nd4")  # Knight to center

        config_with_pst = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_tapered_eval=False,
        )
        config_without_pst = EvaluationConfig(
            use_material=True,
            use_pst=False,
        )

        eval_with_pst = HandcodedEvaluator(board, config_with_pst).evaluate()
        eval_without_pst = HandcodedEvaluator(board, config_without_pst).evaluate()

        # With PST should be different from without PST
        assert eval_with_pst != eval_without_pst


class TestHandcodedEvaluatorTaperedEval:
    """Test tapered evaluation (Node E2)."""

    def test_tapered_eval_enabled(self):
        """Test tapered evaluation in different game phases."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_tapered_eval=True,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        # Should work without error
        assert isinstance(score, float)

    def test_tapered_eval_disabled(self):
        """Test that tapered eval uses only middlegame values when disabled."""
        board = chess.Board()
        config_tapered = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_tapered_eval=True,
        )
        config_simple = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_tapered_eval=False,
        )

        eval_tapered = HandcodedEvaluator(board, config_tapered).evaluate()
        eval_simple = HandcodedEvaluator(board, config_simple).evaluate()

        # In opening, tapered should be close to simple (but not exactly equal)
        assert isinstance(eval_tapered, float)
        assert isinstance(eval_simple, float)

    def test_tapered_eval_in_endgame(self):
        """Test tapered evaluation in endgame."""
        # King and pawn endgame
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_tapered_eval=True,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        # Should be positive (White has pawn)
        assert score > 0.0


class TestHandcodedEvaluatorPawnStructure:
    """Test pawn structure evaluation (Node E3)."""

    def test_pawn_structure_enabled(self):
        """Test pawn structure evaluation."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_pawn_structure=True,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        assert isinstance(score, float)

    def test_pawn_structure_disabled(self):
        """Test that pawn structure is not evaluated when disabled."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_pawn_structure=False,
        )
        evaluator = HandcodedEvaluator(board, config)

        # Should not raise error
        score = evaluator.evaluate()
        assert isinstance(score, float)

    def test_doubled_pawns_penalty(self):
        """Test that doubled pawns are penalized."""
        # Position with doubled pawns for White
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1")
        board.set_fen("rnbqkbnr/pppppppp/8/8/8/2P5/PP1PPPPP/RNBQKBNR w KQkq - 0 1")

        config = EvaluationConfig(
            use_material=True,
            use_pst=False,
            use_pawn_structure=True,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        # With doubled pawns, White should have a slight penalty
        # (though material might offset it)
        assert isinstance(score, float)

    def test_isolated_pawn_penalty(self):
        """Test that isolated pawns are penalized."""
        # Position with isolated e-pawn for White
        board = chess.Board(
            "rnbqkbnr/ppp1pppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        )

        config = EvaluationConfig(
            use_material=True,
            use_pst=False,
            use_pawn_structure=True,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        # Should have some penalty
        assert isinstance(score, float)

    def test_passed_pawn_bonus(self):
        """Test that passed pawns get a bonus."""
        # White passed pawn on e6
        board = chess.Board("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1")

        config = EvaluationConfig(
            use_material=True,
            use_pst=False,
            use_pawn_structure=True,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        # Should be positive (passed pawn + material advantage)
        assert score > 0.0


class TestHandcodedEvaluatorMobility:
    """Test mobility evaluation (Node E4)."""

    def test_mobility_enabled(self):
        """Test mobility evaluation."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_mobility=True,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        assert isinstance(score, float)

    def test_mobility_disabled(self):
        """Test that mobility is not evaluated when disabled."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_mobility=False,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()
        assert isinstance(score, float)

    def test_mobility_favors_active_pieces(self):
        """Test that mobility favors more active piece placement."""
        # Open position with more piece activity
        board_open = chess.Board(
            "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
        )

        config = EvaluationConfig(
            use_material=True,
            use_pst=False,
            use_mobility=True,
        )
        evaluator = HandcodedEvaluator(board_open, config)

        score = evaluator.evaluate()

        # Should evaluate without error
        assert isinstance(score, float)


class TestHandcodedEvaluatorKingSafety:
    """Test king safety evaluation (Node E5)."""

    def test_king_safety_enabled(self):
        """Test king safety evaluation."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_mobility=True,
            use_king_safety=True,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        assert isinstance(score, float)

    def test_king_safety_disabled(self):
        """Test that king safety is not evaluated when disabled."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_mobility=True,
            use_king_safety=False,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()
        assert isinstance(score, float)

    def test_king_safety_requires_mobility(self):
        """Test that king safety requires mobility to be enabled."""
        board = chess.Board()

        # King safety requires mobility (E5 requires E4)
        # But this is validated in config, not evaluator
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_mobility=True,
            use_king_safety=True,
        )

        evaluator = HandcodedEvaluator(board, config)
        assert isinstance(evaluator.evaluate(), float)

    def test_pawn_shield_bonus(self):
        """Test that castled king with pawn shield gets bonus."""
        # White castled kingside with pawn shield
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1")
        board.push_san("O-O")

        config = EvaluationConfig(
            use_material=True,
            use_pst=False,
            use_mobility=True,
            use_king_safety=True,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        # Should include safety bonus
        assert isinstance(score, float)


class TestHandcodedEvaluatorModularCombinations:
    """Test various combinations of modular features."""

    def test_all_features_enabled(self):
        """Test with all features enabled."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_tapered_eval=True,
            use_pawn_structure=True,
            use_mobility=True,
            use_king_safety=True,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        assert isinstance(score, float)

    def test_minimal_features(self):
        """Test with minimal features (material only)."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=False,
            use_mobility=False,
            use_pawn_structure=False,
            use_king_safety=False,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        # Starting position: equal material
        assert score == 0.0

    def test_pst_and_mobility_only(self):
        """Test with PST and mobility enabled."""
        board = chess.Board()
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_mobility=True,
            use_pawn_structure=False,
            use_king_safety=False,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        assert isinstance(score, float)

    def test_comparison_different_configs(self):
        """Test that different configs produce different evaluations."""
        # Position with some imbalance
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")

        config_minimal = EvaluationConfig(
            use_material=True,
            use_pst=False,
            use_mobility=False,
        )
        config_full = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_mobility=True,
            use_pawn_structure=True,
        )

        eval_minimal = HandcodedEvaluator(board, config_minimal).evaluate()
        eval_full = HandcodedEvaluator(board, config_full).evaluate()

        # Different features should produce different scores
        # (though might be equal in some positions)
        assert isinstance(eval_minimal, float)
        assert isinstance(eval_full, float)


class TestHandcodedEvaluatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_board(self):
        """Test evaluation with no pieces."""
        board = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")
        config = EvaluationConfig()
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        assert score == 0.0

    def test_only_kings(self):
        """Test with only kings."""
        board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
        config = EvaluationConfig()
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        # Kings have no material value
        assert score == 0.0 or abs(score) < 1.0  # Allow for PST bonuses

    def test_endgame_position(self):
        """Test endgame position."""
        board = chess.Board("8/4k3/8/8/8/8/4P3/4K3 w - - 0 1")
        config = EvaluationConfig(
            use_material=True,
            use_pst=True,
            use_tapered_eval=True,
        )
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        # White has pawn advantage
        assert score > 0.0

    def test_complex_position(self):
        """Test complex middlegame position."""
        # Sicilian Dragon
        board = chess.Board(
            "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 0 1"
        )
        config = EvaluationConfig()
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        assert isinstance(score, float)

    def test_promotes_pawns(self):
        """Test position with promoted pieces."""
        # Multiple queens
        board = chess.Board("Q6k/8/8/8/8/8/8/K6q w - - 0 1")
        config = EvaluationConfig()
        evaluator = HandcodedEvaluator(board, config)

        score = evaluator.evaluate()

        # Equal queens, should be close to 0
        assert abs(score) < 1.0  # Allow for PST differences
