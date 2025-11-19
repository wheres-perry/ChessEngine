"""
Tests for modular minimax implementation.

These tests verify that the minimax engine can be configured with various
combinations of features, respecting the dependency tree structure.
"""

import pytest

from engine._core import chess_engine_core as chess
from src.engine.config import EngineConfig, SearchConfig
from src.engine.evaluators import MockEvaluator
from src.engine.search.minimax import Minimax


class TestModularMinimaxConfigurations:
    """Test various valid modular configurations of the minimax engine."""

    def test_basic_minimax_only(self):
        """Test pure minimax without any optimizations."""
        cfg = EngineConfig(
            minimax=SearchConfig(
                use_minimax=True,
                use_alpha_beta=False,
                use_iddfs=False,
                use_move_ordering=False,
                use_transposition_table=False,
                use_zobrist=False,
                use_pvs=False,
                use_lmr=False,
                use_quiescence_search=False,
                use_check_extensions=False,
                use_killer_moves=False,
                use_history_heuristic=False,
                use_countermove_heuristic=False,
                use_hash_move_ordering=False,
                use_mvv_lva=False,
                use_see_ordering=False,
                use_tt_aging=False,
                use_iid=False,
                use_delta_pruning=False,
                use_see_pruning_in_qs=False,
                use_null_move_pruning=False,
                use_futility_pruning=False,
                use_extended_futility_pruning=False,
                use_reverse_futility_pruning=False,
                use_aspiration_windows=False,
            )
        )
        board = chess.Board()
        evaluator = MockEvaluator(board)
        engine = Minimax(board, evaluator, cfg)

        # Verify engine was created
        assert engine.use_minimax is True
        assert engine.use_alpha_beta is False
        assert engine.zobrist is None
        assert engine.transposition_table is None
        assert engine.move_orderer is None

        # Should be able to search
        _score, move = engine.find_top_move(depth=1)
        assert move is not None

    def test_minimax_with_alpha_beta(self):
        """Test minimax with alpha-beta pruning."""
        cfg = EngineConfig(
            minimax=SearchConfig(
                use_minimax=True,
                use_alpha_beta=True,
                use_iddfs=False,
                use_move_ordering=False,
                use_transposition_table=False,
                use_zobrist=False,
                use_pvs=False,
                use_lmr=False,
                use_quiescence_search=False,
                use_check_extensions=False,
                use_killer_moves=False,
                use_history_heuristic=False,
                use_countermove_heuristic=False,
                use_hash_move_ordering=False,
                use_mvv_lva=False,
                use_see_ordering=False,
                use_tt_aging=False,
                use_iid=False,
                use_delta_pruning=False,
                use_see_pruning_in_qs=False,
                use_null_move_pruning=False,
                use_futility_pruning=False,
                use_extended_futility_pruning=False,
                use_reverse_futility_pruning=False,
                use_aspiration_windows=False,
            )
        )
        board = chess.Board()
        evaluator = MockEvaluator(board)
        engine = Minimax(board, evaluator, cfg)

        assert engine.use_minimax is True
        assert engine.use_alpha_beta is True
        assert engine.use_pvs is False

    def test_minimax_with_transposition_table(self):
        """Test minimax with TT and Zobrist (no alpha-beta)."""
        cfg = EngineConfig(
            minimax=SearchConfig(
                use_minimax=True,
                use_alpha_beta=False,
                use_iddfs=False,
                use_move_ordering=False,
                use_transposition_table=True,
                use_zobrist=True,
                use_pvs=False,
                use_lmr=False,
                use_quiescence_search=False,
                use_check_extensions=False,
                use_killer_moves=False,
                use_history_heuristic=False,
                use_countermove_heuristic=False,
                use_hash_move_ordering=False,
                use_mvv_lva=False,
                use_see_ordering=False,
                use_tt_aging=False,
                use_iid=False,
                use_delta_pruning=False,
                use_see_pruning_in_qs=False,
                use_null_move_pruning=False,
                use_futility_pruning=False,
                use_extended_futility_pruning=False,
                use_reverse_futility_pruning=False,
                use_aspiration_windows=False,
            )
        )
        board = chess.Board()
        evaluator = MockEvaluator(board)
        engine = Minimax(board, evaluator, cfg)

        assert engine.use_minimax is True
        assert engine.use_transposition_table is True
        assert engine.use_zobrist is True
        assert engine.zobrist is not None
        assert engine.transposition_table is not None

    def test_full_featured_engine(self):
        """Test engine with all features enabled (default config)."""
        cfg = EngineConfig()
        board = chess.Board()
        evaluator = MockEvaluator(board)
        engine = Minimax(board, evaluator, cfg)

        # All major features should be enabled
        assert engine.use_minimax is True
        assert engine.use_alpha_beta is True
        assert engine.use_iddfs is True
        assert engine.use_move_ordering is True
        assert engine.use_transposition_table is True
        assert engine.use_zobrist is True
        assert engine.use_pvs is True

        # Components should be initialized
        assert engine.zobrist is not None
        assert engine.transposition_table is not None
        assert engine.move_orderer is not None


class TestModularDependencyValidation:
    """Test that invalid configurations are rejected."""

    def test_alpha_beta_without_minimax_rejected(self):
        """Test that alpha-beta requires minimax."""
        with pytest.raises(
            ValueError, match="All search optimizations require basic minimax"
        ):
            EngineConfig(
                minimax=SearchConfig(
                    use_minimax=False,
                    use_alpha_beta=True,
                )
            )

    def test_pvs_without_alpha_beta_rejected(self):
        """Test that PVS requires alpha-beta."""
        with pytest.raises(ValueError, match="Principal Variation Search"):
            EngineConfig(
                minimax=SearchConfig(
                    use_minimax=True,
                    use_alpha_beta=False,
                    use_iddfs=True,
                    use_pvs=True,
                    use_transposition_table=False,
                    use_zobrist=False,
                    use_lmr=False,
                    use_quiescence_search=False,
                )
            )

    def test_lmr_without_move_ordering_rejected(self):
        """Test that LMR requires both alpha-beta and move ordering."""
        with pytest.raises(ValueError, match="Late Move Reduction"):
            EngineConfig(
                minimax=SearchConfig(
                    use_minimax=True,
                    use_alpha_beta=True,
                    use_move_ordering=False,
                    use_lmr=True,
                    use_transposition_table=False,
                    use_zobrist=False,
                    use_pvs=False,
                    use_quiescence_search=False,
                    use_killer_moves=False,  # Requires move ordering
                    use_history_heuristic=False,  # Requires move ordering
                    use_countermove_heuristic=False,  # Requires move ordering
                    use_mvv_lva=False,  # Requires move ordering
                    use_see_ordering=False,  # Requires move ordering
                    use_hash_move_ordering=False,  # Requires TT
                )
            )

    def test_tt_without_zobrist_rejected(self):
        """Test that transposition table requires Zobrist hashing."""
        with pytest.raises(ValueError, match="Transposition table requires Zobrist"):
            EngineConfig(
                minimax=SearchConfig(
                    use_minimax=True,
                    use_transposition_table=True,
                    use_zobrist=False,
                )
            )

    def test_zobrist_without_tt_rejected(self):
        """Test that Zobrist should only be enabled with TT."""
        with pytest.raises(ValueError, match="Zobrist hashing is only useful"):
            EngineConfig(
                minimax=SearchConfig(
                    use_minimax=True,
                    use_zobrist=True,
                    use_transposition_table=False,
                )
            )

    def test_iid_without_tt_rejected(self):
        """Test that IID requires both IDDFS and TT."""
        with pytest.raises(ValueError, match="Internal Iterative Deepening"):
            EngineConfig(
                minimax=SearchConfig(
                    use_minimax=True,
                    use_iddfs=True,
                    use_iid=True,
                    use_transposition_table=False,
                    use_zobrist=False,
                    use_tt_aging=False,
                    use_hash_move_ordering=False,  # Requires TT
                )
            )

    def test_killer_moves_without_alpha_beta_rejected(self):
        """Test that killer moves require both move ordering and alpha-beta."""
        with pytest.raises(ValueError, match="Killer heuristic"):
            EngineConfig(
                minimax=SearchConfig(
                    use_minimax=True,
                    use_alpha_beta=False,
                    use_move_ordering=True,
                    use_killer_moves=True,
                    use_transposition_table=False,
                    use_zobrist=False,
                    use_lmr=False,
                    use_pvs=False,
                    use_quiescence_search=False,
                    use_null_move_pruning=False,  # Requires alpha-beta
                    use_futility_pruning=False,  # Requires alpha-beta
                    use_extended_futility_pruning=False,  # Requires alpha-beta
                    use_reverse_futility_pruning=False,  # Requires alpha-beta
                    use_aspiration_windows=False,  # Requires alpha-beta
                    use_history_heuristic=False,  # Also requires alpha-beta
                    use_countermove_heuristic=False,  # Also requires alpha-beta
                )
            )
