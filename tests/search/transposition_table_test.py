"""
Test suite for the TranspositionTable chess position cache.

This module contains tests for the TranspositionTable class, which provides
position caching to avoid recalculating previously evaluated positions.
Tests cover:
- Basic storage and retrieval
- Aging mechanism for table entries
- Entry type classification (exact, upper, lower bounds)
- Size management and replacement policies
- Integration with Minimax search
"""

from unittest.mock import MagicMock, patch

import chess
import pytest

from src.engine.config import EngineConfig, MinimaxConfig
from src.engine.evaluators.mock_eval import MockEval
from src.engine.search.minimax import Minimax
from src.engine.search.transposition_table import TranspositionTable


class TestTranspositionTableBasics:
    """Test basic functionality of TranspositionTable."""

    def test_store_and_lookup(self):
        """Test storing and retrieving entries."""
        tt = TranspositionTable()
        hash_val = 12345
        depth = 3
        score = 1.5
        alpha = 1.0
        beta = 2.0

        # Store entry

        tt.store(hash_val, depth, score, alpha, beta, alpha)

        # Lookup with sufficient depth

        result = tt.lookup(hash_val, depth, alpha, beta)
        assert result == score

        # Lookup with higher depth (should return None)

        result = tt.lookup(hash_val, depth + 1, alpha, beta)
        assert result is None

    def test_clear(self):
        """Test clearing the table."""
        tt = TranspositionTable()

        # Store some entries

        tt.store(1, 3, 0.5, 0, 1, 0)
        tt.store(2, 3, -0.5, 0, 1, 0)

        assert tt.size() == 2

        # Clear table

        tt.clear()
        assert tt.size() == 0

    def test_size_limit(self):
        """Test that the table respects its size limit."""
        max_entries = 10
        tt = TranspositionTable(max_entries=max_entries)

        # Fill up the table

        for i in range(max_entries + 5):
            tt.store(i, 3, float(i), 0, float(i + 1), 0)
        assert tt.size() <= max_entries


class TestTranspositionTableAging:
    """Test aging mechanism of TranspositionTable."""

    def test_aging_enabled(self):
        """Test that entries age with each increment."""
        tt = TranspositionTable(use_tt_aging=True)
        hash_val = 12345

        # Store entry at age 0

        tt.store(hash_val, 3, 1.5, 1.0, 2.0, 1.0)

        # Lookup should succeed

        assert tt.lookup(hash_val, 3, 1.0, 2.0) == 1.5

        # Increment age multiple times to exceed MAX_AGE_DIFF

        for _ in range(tt.MAX_AGE_DIFF + 1):
            tt.increment_age()
        # Now the entry should be too old

        assert tt.lookup(hash_val, 3, 1.0, 2.0) is None

    def test_aging_disabled(self):
        """Test that entries don't age when aging is disabled."""
        tt = TranspositionTable(use_tt_aging=False)
        hash_val = 12345

        # Store entry

        tt.store(hash_val, 3, 1.5, 1.0, 2.0, 1.0)

        # Increment age multiple times - shouldn't matter

        for _ in range(10):
            tt.increment_age()
        # Lookup should still succeed

        assert tt.lookup(hash_val, 3, 1.0, 2.0) == 1.5

    def test_entry_refresh(self):
        """Test that storing again refreshes an entry's age."""
        tt = TranspositionTable(use_tt_aging=True)
        hash_val = 12345

        # Store entry at age 0

        tt.store(hash_val, 3, 1.5, 1.0, 2.0, 1.0)

        # Increment age a few times

        for _ in range(tt.MAX_AGE_DIFF):
            tt.increment_age()
        # Entry should still be valid but getting old

        assert tt.lookup(hash_val, 3, 1.0, 2.0) == 1.5

        # Store again to refresh the age

        tt.store(hash_val, 3, 1.5, 1.0, 2.0, 1.0)

        # Increment age again to exceed the original MAX_AGE_DIFF

        tt.increment_age()

        # Entry should still be valid because we refreshed it

        assert tt.lookup(hash_val, 3, 1.0, 2.0) == 1.5

    def test_age_reset(self):
        """Test that age can be reset."""
        tt = TranspositionTable(use_tt_aging=True)
        hash_val = 12345

        # Store entry

        tt.store(hash_val, 3, 1.5, 1.0, 2.0, 1.0)

        # Increment age multiple times

        for _ in range(tt.MAX_AGE_DIFF):
            tt.increment_age()
        # Reset age

        tt.reset_age()

        # Entry should now be too old relative to the reset age

        assert tt.lookup(hash_val, 3, 1.0, 2.0) is None


class TestTranspositionTableEntryTypes:
    """Test different entry types (exact, upper, lower)."""

    def test_exact_score(self):
        """Test exact score entries."""
        tt = TranspositionTable()
        hash_val = 12345

        # Store exact value (between alpha and beta)

        score = 1.5
        alpha = 1.0
        beta = 2.0
        tt.store(hash_val, 3, score, alpha, beta, alpha)

        # Should return exact score

        assert tt.lookup(hash_val, 3, 1.0, 2.0) == score

    def test_upper_bound(self):
        """Test upper bound entries."""
        tt = TranspositionTable()
        hash_val = 12345

        # Store upper bound (score <= alpha)

        score = 0.5
        alpha = 1.0
        beta = 2.0
        tt.store(hash_val, 3, score, alpha, beta, alpha)

        # Should return alpha when score <= alpha

        assert tt.lookup(hash_val, 3, alpha, beta) == alpha

    def test_lower_bound(self):
        """Test lower bound entries."""
        tt = TranspositionTable()
        hash_val = 12345

        # Store lower bound (score >= beta)

        score = 2.5
        alpha = 1.0
        beta = 2.0
        tt.store(hash_val, 3, score, alpha, beta, alpha)

        # Should return beta when score >= beta

        assert tt.lookup(hash_val, 3, alpha, beta) == beta


class TestTranspositionTableIntegration:
    """Test TranspositionTable integration with Minimax search."""

    def test_config_validation(self):
        """Test that config validation catches invalid TT aging configuration."""
        board = chess.Board()
        evaluator = MockEval(board)

        # Attempt to enable aging without Zobrist (should raise ValueError)

        with pytest.raises(
            ValueError, match="Transposition table aging requires Zobrist hashing"
        ):
            config = EngineConfig(
                minimax=MinimaxConfig(use_zobrist=False, use_tt_aging=True)
            )
            Minimax(board, evaluator, config)

    def test_node_count_reduction(self):
        """Test that TT reduces node count during search."""
        board = chess.Board()
        evaluator = MockEval(board)
        depth = 5

        # First search without TT

        config_no_tt = EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=False,
                use_tt_aging=False,
                max_time=None,  # No timeout for testing
            )
        )
        minimax_no_tt = Minimax(board, evaluator, config_no_tt)
        minimax_no_tt.find_top_move(depth=depth)
        nodes_without_tt = minimax_no_tt.node_count

        # Then search with TT

        config_with_tt = EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=True,
                use_tt_aging=True,
                max_time=None,  # No timeout for testing
            )
        )
        minimax_with_tt = Minimax(board, evaluator, config_with_tt)
        minimax_with_tt.find_top_move(depth=depth)
        nodes_with_tt = minimax_with_tt.node_count

        # Should see a significant reduction in node count

        assert nodes_with_tt < nodes_without_tt

        # Expect reasonable reduction (at least 25%)

        reduction_ratio = (nodes_without_tt - nodes_with_tt) / nodes_without_tt
        assert reduction_ratio > 0.25, f"Node reduction only {reduction_ratio:.2%}"

    def test_aging_effectiveness(self):
        """Test that aging mechanism correctly manages TT entries across searches."""
        board = chess.Board()
        evaluator = MockEval(board)
        depth = 5

        config = EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=True,
                use_tt_aging=True,
                max_time=None,  # No timeout for testing
            )
        )
        minimax = Minimax(board, evaluator, config)

        # Do an initial search

        minimax.find_top_move(depth=depth)
        tt_size_after_first = minimax.transposition_table.size()
        assert tt_size_after_first > 0, "TT should not be empty after search"

        # Make a few moves to create different positions

        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        board.push_san("Nc6")

        # Do a second search

        minimax.find_top_move(depth=depth)
        tt_size_after_second = minimax.transposition_table.size()

        # Entries should be kept and more added

        assert tt_size_after_second >= tt_size_after_first

        # Now make many more searches with different positions to force aging

        for _ in range(5):
            if board.turn == chess.WHITE:
                # Try some common legal white moves

                for candidate_move in ["Bc4", "d4", "Nc3", "Nf3", "Qe2", "0-0", "h3"]:
                    try:
                        board.push_san(candidate_move)
                        break  # Found a legal move
                    except chess.IllegalMoveError:
                        continue
                else:
                    # If we get here, none of our candidates worked - just make any legal move

                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(legal_moves[0])
            else:
                # Try some common legal black moves

                for candidate_move in ["Nf6", "d5", "e6", "Bc5", "0-0", "h6"]:
                    try:
                        board.push_san(candidate_move)
                        break  # Found a legal move
                    except chess.IllegalMoveError:
                        continue
                else:
                    # If we get here, just make any legal move

                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(legal_moves[0])
            minimax.find_top_move(depth=depth)
        # TT shouldn't grow unbounded due to aging and replacement

        assert (
            minimax.transposition_table.size()
            <= minimax.transposition_table.max_entries
        )


if __name__ == "__main__":
    pytest.main([__file__])
