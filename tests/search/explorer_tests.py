"""
Comprehensive tests for Explorer search engine.

Tests cover:
- Modular control flow based on configuration
- Pure minimax vs alpha-beta
- Iterative deepening
- Transposition table integration
- Move ordering integration
- Configuration validation
"""

import pytest

from engine._core import chess_engine_core as chess
from src.engine.config import EngineConfig, SearchConfig
from src.engine.evaluators import MockEvaluator, SimpleEvaluator
from src.engine.search.explorer import Explorer


class TestExplorerInitialization:
    """Test Explorer initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)
        config = EngineConfig()

        explorer = Explorer(board, evaluator, config)

        assert explorer.board == board
        assert explorer.evaluator == evaluator
        assert explorer.use_minimax is True

    def test_init_with_minimal_config(self):
        """Test initialization with minimal configuration."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)
        config = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=False,
                use_iddfs=False,
                use_move_ordering=False,
                use_transposition_table=False,
                use_zobrist=False,
            )
        )

        explorer = Explorer(board, evaluator, config)

        assert explorer.use_minimax is True
        assert explorer.use_alpha_beta is False
        assert explorer.use_iddfs is False
        assert explorer.zobrist is None
        assert explorer.transposition_table is None
        assert explorer.move_orderer is None

    def test_init_with_full_config(self):
        """Test initialization with full configuration."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)
        config = EngineConfig()  # Default enables all features

        explorer = Explorer(board, evaluator, config)

        assert explorer.use_minimax is True
        assert explorer.use_alpha_beta is True
        assert explorer.use_iddfs is True
        assert explorer.zobrist is not None
        assert explorer.transposition_table is not None
        assert explorer.move_orderer is not None


class TestExplorerBasicSearch:
    """Test basic search functionality."""

    def test_search_returns_move(self):
        """Test that search returns a valid move."""
        board = chess.Board()
        evaluator = SimpleEvaluator(board, None)
        config = EngineConfig()

        explorer = Explorer(board, evaluator, config)
        score, move = explorer.search(depth=1)

        assert move is not None
        assert move in board.legal_moves
        assert isinstance(score, (int, float))

    def test_search_with_minimax_disabled(self):
        """Test that search returns None when minimax is disabled."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)
        config = EngineConfig(
            search=SearchConfig(
                use_minimax=False,
                use_alpha_beta=False,
                use_iddfs=False,
                use_move_ordering=False,
                use_transposition_table=False,
                use_zobrist=False,
            )
        )

        explorer = Explorer(board, evaluator, config)
        score, move = explorer.search(depth=1)

        # Should return None when minimax disabled
        assert score is None
        assert move is None

    def test_search_depth_1(self):
        """Test search at depth 1."""
        board = chess.Board()
        evaluator = SimpleEvaluator(board, None)
        config = EngineConfig()

        explorer = Explorer(board, evaluator, config)
        score, move = explorer.search(depth=1)

        assert move is not None
        assert isinstance(score, float)

    def test_search_increments_node_count(self):
        """Test that node count is incremented during search."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)
        config = EngineConfig()

        explorer = Explorer(board, evaluator, config)
        explorer.search(depth=2)

        assert explorer.nodes_searched > 0


class TestExplorerPureMinimaxMode:
    """Test Explorer in pure minimax mode (no alpha-beta)."""

    def test_pure_minimax_enabled(self):
        """Test pure minimax without alpha-beta."""
        board = chess.Board()
        evaluator = SimpleEvaluator(board, None)
        config = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=False,
                use_iddfs=False,
                use_move_ordering=False,
                use_transposition_table=False,
                use_zobrist=False,
            )
        )

        explorer = Explorer(board, evaluator, config)
        _score, move = explorer.search(depth=2)

        assert move is not None
        assert explorer.use_alpha_beta is False

    def test_pure_minimax_explores_all_branches(self):
        """Test that pure minimax explores more nodes than alpha-beta."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)

        config_pure = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=False,
                use_iddfs=False,
                use_move_ordering=False,
                use_transposition_table=False,
                use_zobrist=False,
            )
        )
        config_ab = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=True,
                use_iddfs=False,
                use_move_ordering=False,
                use_transposition_table=False,
                use_zobrist=False,
            )
        )

        explorer_pure = Explorer(board, evaluator, config_pure)
        explorer_ab = Explorer(board, evaluator, config_ab)

        explorer_pure.search(depth=3)
        explorer_ab.search(depth=3)

        # Pure minimax should search more nodes (no pruning)
        assert explorer_pure.nodes_searched >= explorer_ab.nodes_searched


class TestExplorerAlphaBetaMode:
    """Test Explorer with alpha-beta pruning."""

    def test_alpha_beta_enabled(self):
        """Test alpha-beta pruning mode."""
        board = chess.Board()
        evaluator = SimpleEvaluator(board, None)
        config = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=True,
                use_iddfs=False,
                use_move_ordering=False,
                use_transposition_table=False,
                use_zobrist=False,
            )
        )

        explorer = Explorer(board, evaluator, config)
        _score, move = explorer.search(depth=2)

        assert move is not None
        assert explorer.use_alpha_beta is True

    def test_alpha_beta_prunes_branches(self):
        """Test that alpha-beta prunes some branches."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)
        config = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=True,
                use_iddfs=False,
                use_move_ordering=False,
                use_transposition_table=False,
                use_zobrist=False,
            )
        )

        explorer = Explorer(board, evaluator, config)
        explorer.search(depth=3)

        # Should search fewer nodes than total possible
        # At depth 3 from start: ~20 * 20 * 20 = 8000 nodes without pruning
        # With pruning should be less
        assert explorer.nodes_searched < 8000


class TestExplorerIterativeDeepening:
    """Test Explorer with iterative deepening."""

    def test_iddfs_enabled(self):
        """Test iterative deepening search."""
        board = chess.Board()
        evaluator = SimpleEvaluator(board, None)
        config = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=True,
                use_iddfs=True,
                use_move_ordering=False,
                use_transposition_table=False,
                use_zobrist=False,
            )
        )

        explorer = Explorer(board, evaluator, config)
        _score, move = explorer.search(depth=3)

        assert move is not None
        assert explorer.use_iddfs is True

    def test_iddfs_disabled_uses_fixed_depth(self):
        """Test that fixed depth search is used when IDDFS disabled."""
        board = chess.Board()
        evaluator = SimpleEvaluator(board, None)
        config = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=True,
                use_iddfs=False,
                use_move_ordering=False,
                use_transposition_table=False,
                use_zobrist=False,
            )
        )

        explorer = Explorer(board, evaluator, config)
        _score, move = explorer.search(depth=2)

        assert move is not None
        assert explorer.use_iddfs is False

    def test_iddfs_depth_1_uses_fixed_depth(self):
        """Test that depth 1 uses fixed depth even with IDDFS enabled."""
        board = chess.Board()
        evaluator = SimpleEvaluator(board, None)
        config = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_iddfs=True,
            )
        )

        explorer = Explorer(board, evaluator, config)
        _score, move = explorer.search(depth=1)

        # Should still work at depth 1
        assert move is not None


class TestExplorerTranspositionTable:
    """Test Explorer with transposition table."""

    def test_tt_enabled(self):
        """Test with transposition table enabled."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)
        config = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=True,
                use_zobrist=True,
                use_transposition_table=True,
            )
        )

        explorer = Explorer(board, evaluator, config)

        assert explorer.use_transposition_table is True
        assert explorer.zobrist is not None
        assert explorer.transposition_table is not None

    def test_tt_reduces_node_count(self):
        """Test that TT reduces node count."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)

        config_no_tt = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=True,
                use_zobrist=False,
                use_transposition_table=False,
                use_iid=False,
            )
        )
        config_with_tt = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=True,
                use_zobrist=True,
                use_transposition_table=True,
            )
        )

        explorer_no_tt = Explorer(board, evaluator, config_no_tt)
        explorer_with_tt = Explorer(board, evaluator, config_with_tt)

        explorer_no_tt.search(depth=4)
        explorer_with_tt.search(depth=4)

        # TT should reduce node count
        assert explorer_with_tt.nodes_searched < explorer_no_tt.nodes_searched


class TestExplorerMoveOrdering:
    """Test Explorer with move ordering."""

    def test_move_ordering_enabled(self):
        """Test with move ordering enabled."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)
        config = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=True,
                use_move_ordering=True,
                use_mvv_lva=True,
            )
        )

        explorer = Explorer(board, evaluator, config)

        assert explorer.use_move_ordering is True
        assert explorer.move_orderer is not None

    def test_move_ordering_disabled(self):
        """Test with move ordering disabled."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)
        config = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=True,
                use_move_ordering=False,
                use_transposition_table=False,
                use_zobrist=False,
            )
        )

        explorer = Explorer(board, evaluator, config)

        assert explorer.use_move_ordering is False
        assert explorer.move_orderer is None

    def test_move_ordering_improves_pruning(self):
        """Test that move ordering improves alpha-beta pruning."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)

        config_no_ordering = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=True,
                use_move_ordering=False,
                use_transposition_table=False,
                use_zobrist=False,
            )
        )
        config_with_ordering = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=True,
                use_move_ordering=True,
                use_mvv_lva=True,
                use_transposition_table=False,
                use_zobrist=False,
                use_killer_moves=False,
            )
        )

        explorer_no_ordering = Explorer(board, evaluator, config_no_ordering)
        explorer_with_ordering = Explorer(board, evaluator, config_with_ordering)

        explorer_no_ordering.search(depth=3)
        explorer_with_ordering.search(depth=3)

        # Move ordering should reduce node count (better pruning)
        assert (
            explorer_with_ordering.nodes_searched <= explorer_no_ordering.nodes_searched
        )


class TestExplorerEdgeCases:
    """Test edge cases and error conditions."""

    def test_search_from_checkmate_position(self):
        """Test search from checkmate position."""
        # Fool's mate
        board = chess.Board()
        board.push_san("f3")
        board.push_san("e5")
        board.push_san("g4")
        board.push_san("Qh4#")

        evaluator = SimpleEvaluator(board, None)
        config = EngineConfig()

        explorer = Explorer(board, evaluator, config)
        _score, move = explorer.search(depth=1)

        # No legal moves in checkmate
        assert move is None

    def test_search_from_stalemate_position(self):
        """Test search from stalemate position."""
        board = chess.Board.from_fen("7k/8/8/8/8/8/8/K6Q b - - 0 1")

        evaluator = SimpleEvaluator(board, None)
        config = EngineConfig()

        explorer = Explorer(board, evaluator, config)
        _score, move = explorer.search(depth=1)

        # No legal moves in stalemate
        assert move is None

    def test_search_with_only_one_legal_move(self):
        """Test search when only one legal move available."""
        # Position with only one legal move
        board = chess.Board.from_fen("7k/8/5K2/8/8/8/7R/8 b - - 0 1")

        evaluator = SimpleEvaluator(board, None)
        config = EngineConfig()

        explorer = Explorer(board, evaluator, config)
        _score, move = explorer.search(depth=1)

        # Should find the only legal move
        assert move is not None
        assert len(list(board.legal_moves)) == 1

    def test_search_depth_0(self):
        """Test search at depth 0."""
        board = chess.Board()
        evaluator = SimpleEvaluator(board, None)
        config = EngineConfig()

        explorer = Explorer(board, evaluator, config)
        score, _move = explorer.search(depth=0)

        # Depth 0 should return evaluation, not search
        # Implementation dependent
        assert isinstance(score, (int, float, type(None)))


class TestExplorerConsistency:
    """Test consistency across different configurations."""

    def test_same_position_same_result(self):
        """Test that same position gives same result."""
        board = chess.Board()
        evaluator = SimpleEvaluator(board, None)
        config = EngineConfig()

        explorer1 = Explorer(board, evaluator, config)
        explorer2 = Explorer(board, evaluator, config)

        score1, move1 = explorer1.search(depth=2)
        score2, move2 = explorer2.search(depth=2)

        # Should get same score
        assert score1 == score2
        # Move might differ if multiple moves have same evaluation
        # but both should be legal
        assert move1 is not None
        assert move2 is not None
        assert move1 in board.legal_moves
        assert move2 in board.legal_moves

    def test_configuration_affects_search(self):
        """Test that different configurations affect search."""
        board = chess.Board()
        evaluator = MockEvaluator(board, None)

        config1 = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=False,
            )
        )
        config2 = EngineConfig(
            search=SearchConfig(
                use_minimax=True,
                use_alpha_beta=True,
            )
        )

        explorer1 = Explorer(board, evaluator, config1)
        explorer2 = Explorer(board, evaluator, config2)

        explorer1.search(depth=3)
        explorer2.search(depth=3)

        # Different configs should affect node count
        assert explorer1.nodes_searched != explorer2.nodes_searched
