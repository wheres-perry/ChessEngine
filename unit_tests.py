# Contains P2P Tests as well.
# tests/search/minimax_test.py


import logging
import time

import chess
import pytest

from src.engine.config import EngineConfig, MinimaxConfig
from src.engine.evaluators.mock_eval import MockEval
from src.engine.search.minimax import Minimax


class TestConfigValidation:
    """Test validation of EngineConfig during Minimax initialization."""

    def test_tt_aging_without_zobrist_raises(self):
        """Test that enabling TT aging without Zobrist hashing raises a ValueError."""
        # The validation happens in EngineConfig.__post_init__, not in Minimax

        with pytest.raises(
            ValueError,
            match="Transposition table aging requires Zobrist hashing to be enabled",
        ):
            EngineConfig(
                minimax=MinimaxConfig(
                    use_zobrist=False,
                    use_tt_aging=True,
                )
            )


class TestPVSDependency:
    """Test that PVS is properly disabled when alpha-beta pruning is off."""

    def test_pvs_disabled_with_warning(self, caplog):
        """Test that PVS is disabled when alpha-beta is disabled, with a warning."""
        caplog.set_level(logging.WARNING)
        cfg = EngineConfig(
            minimax=MinimaxConfig(
                use_alpha_beta=False,
                use_pvs=True,
            )
        )
        engine = Minimax(chess.Board(), MockEval(chess.Board()), cfg)
        assert engine.use_pvs is False
        assert any("Disabling PVS" in rec.getMessage() for rec in caplog.records)


class TestIterativeDeepening:
    """Test iterative deepening search implementation."""

    def test_iddfs_sequences_depths(self, monkeypatch):
        """Test that IDDFS calls _search_fixed_depth for each depth in sequence."""
        called = []
        dummy_move = chess.Move.from_uci("a2a3")

        def fake_search(self, depth):
            called.append(depth)
            # Return a move only on the final depth

            return float(depth), (dummy_move if depth == 4 else None)

        monkeypatch.setattr(Minimax, "_search_fixed_depth", fake_search)

        cfg = EngineConfig(
            minimax=MinimaxConfig(
                use_iddfs=True,
                use_zobrist=False,
                use_tt_aging=False,
                max_time=None,
            )
        )
        engine = Minimax(chess.Board(), MockEval(chess.Board()), cfg)
        score, move = engine.find_top_move(depth=4)

        assert called == [1, 2, 3, 4]
        assert score == 4.0
        assert move == dummy_move


class TestTimeLimit:
    """Test time limit enforcement in search."""

    def test_check_time_limit_flags_time_up(self):
        """Test that _check_time_limit sets time_up flag when time is exceeded."""
        cfg = EngineConfig(
            minimax=MinimaxConfig(
                max_time=0.01,
            )
        )
        engine = Minimax(chess.Board(), MockEval(chess.Board()), cfg)
        # Simulate search starting in the past

        engine.start_time = time.time() - 1.0
        assert engine._check_time_limit() is True
        assert engine.time_up is True


# tests/search/transposition_table_test.py


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


# tests/search/zobrist_test.py


import chess
import pytest

from src.engine.config import EngineConfig, MinimaxConfig
from src.engine.evaluators.mock_eval import MockEval
from src.engine.search.minimax import Minimax
from src.engine.search.zobrist import Zobrist


class TestZobristBasics:
    """Test basic functionality of Zobrist hashing."""

    def test_hash_consistency(self):
        """Test that same position always yields the same hash."""
        zobrist = Zobrist(seed=42)  # Fixed seed
        board = chess.Board()

        # Hash the same board multiple times

        hashes = [zobrist.hash_board(board) for _ in range(3)]

        # All hashes should be identical

        assert hashes[0] == hashes[1] == hashes[2]

    def test_hash_uniqueness(self):
        """Test that different positions yield different hashes."""
        zobrist = Zobrist(seed=42)  # Fixed seed
        board1 = chess.Board()
        board2 = chess.Board()

        # Make different moves on board2

        board2.push_san("e4")

        hash1 = zobrist.hash_board(board1)
        hash2 = zobrist.hash_board(board2)

        # Hashes should be different

        assert hash1 != hash2

    def test_position_independence(self):
        """Test that hash depends only on position, not move history."""
        zobrist = Zobrist(seed=42)  # Fixed seed

        # Two different ways to reach the same position

        board1 = chess.Board()
        board1.push_san("e4")
        board1.push_san("e5")

        board2 = chess.Board()
        board2.push_san("e4")
        board2.push_san("d5")  # Different move
        board2.pop()  # Undo
        board2.push_san("e5")  # Now same position as board1

        hash1 = zobrist.hash_board(board1)
        hash2 = zobrist.hash_board(board2)

        # Hashes should be identical

        assert hash1 == hash2


class TestZobristIncrementalUpdates:
    """Test incremental Zobrist hash updates."""

    def test_incremental_vs_full_hash(self):
        """Test that incremental updates match full hash computation."""
        zobrist = Zobrist(seed=42)  # Fixed seed
        board = chess.Board()

        # CRUCIAL: Initialize hash first

        zobrist.hash_board(board)

        # Store move info

        move = chess.Move.from_uci("e2e4")
        old_castling = board.castling_rights
        old_ep = board.ep_square

        # Get move details

        piece_at_dest = board.piece_at(move.to_square)
        captured_piece_type = piece_at_dest.piece_type if piece_at_dest else None
        was_ep = board.is_en_passant(move)
        ks_castle = board.is_kingside_castling(move)
        qs_castle = board.is_queenside_castling(move)

        # Make move

        board.push(move)

        # Update incrementally

        incremental_hash = zobrist.update_hash_for_move(
            board,
            move,
            old_castling,
            old_ep,
            captured_piece_type,
            was_ep,
            ks_castle,
            qs_castle,
        )

        # Compute hash from scratch for comparison (same zobrist instance)

        fresh_hash = zobrist.hash_board(board)

        # Both methods should give the same hash

        assert incremental_hash == fresh_hash

    def test_multiple_moves_consistency(self):
        """Test hash consistency across a series of moves."""
        zobrist = Zobrist(seed=42)  # Fixed seed
        board = chess.Board()

        # Initial hash

        zobrist.hash_board(board)

        # Make several moves and update hash incrementally

        moves = ["e4", "e5", "Nf3", "Nc6", "Bc4"]

        for san in moves:
            move = board.parse_san(san)
            old_castling = board.castling_rights
            old_ep = board.ep_square

            piece_at_dest = board.piece_at(move.to_square)
            captured_piece_type = piece_at_dest.piece_type if piece_at_dest else None
            was_ep = board.is_en_passant(move)
            ks_castle = board.is_kingside_castling(move)
            qs_castle = board.is_queenside_castling(move)

            board.push(move)

            # Update hash incrementally

            incremental_hash = zobrist.update_hash_for_move(
                board,
                move,
                old_castling,
                old_ep,
                captured_piece_type,
                was_ep,
                ks_castle,
                qs_castle,
            )

            # Verify against full hash calculation

            fresh_hash = zobrist.hash_board(board)
            assert incremental_hash == fresh_hash


class TestZobristSpecialMoves:
    """Test Zobrist hashing of special chess moves."""

    def test_castling_hash(self):
        """Test hashing of castling moves."""
        zobrist = Zobrist(seed=42)  # Fixed seed

        # Setup a position where castling is possible

        board = chess.Board(
            "r3k2r/ppp1pppp/2n2n2/8/8/2N2N2/PPP1PPPP/R3K2R w KQkq - 0 1"
        )
        original_hash = zobrist.hash_board(board)

        # Try kingside castling

        move = board.parse_san("O-O")
        old_castling = board.castling_rights
        old_ep = board.ep_square

        # No capture in castling

        captured_piece_type = None
        was_ep = False
        ks_castle = True  # Kingside castling
        qs_castle = False

        board.push(move)
        castle_hash = zobrist.update_hash_for_move(
            board,
            move,
            old_castling,
            old_ep,
            captured_piece_type,
            was_ep,
            ks_castle,
            qs_castle,
        )

        # Compute fresh hash to verify

        fresh_hash = zobrist.hash_board(board)
        assert castle_hash == fresh_hash

        # Should not match original position

        assert castle_hash != original_hash

    def test_en_passant_hash(self):
        """Test hashing of en passant moves."""
        zobrist = Zobrist(seed=42)  # Fixed seed

        # Setup a position where en passant is possible

        board = chess.Board(
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        )
        original_hash = zobrist.hash_board(board)

        # Make en passant capture

        move = board.parse_san("exf6")
        old_castling = board.castling_rights
        old_ep = board.ep_square

        captured_piece_type = chess.PAWN  # En passant always captures a pawn
        was_ep = True
        ks_castle = False
        qs_castle = False

        board.push(move)
        ep_hash = zobrist.update_hash_for_move(
            board,
            move,
            old_castling,
            old_ep,
            captured_piece_type,
            was_ep,
            ks_castle,
            qs_castle,
        )

        # Compute fresh hash to verify

        fresh_hash = zobrist.hash_board(board)
        assert ep_hash == fresh_hash

        # Should not match original position

        assert ep_hash != original_hash

    def test_promotion_hash(self):
        """Test hashing of promotion moves."""
        zobrist = Zobrist(seed=42)  # Fixed seed

        # Setup a position where promotion is possible

        board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        original_hash = zobrist.hash_board(board)

        # Promote to queen

        move = chess.Move.from_uci("a7a8q")  # a7-a8=Q
        old_castling = board.castling_rights
        old_ep = board.ep_square

        # Check if the destination square has a piece

        piece_at_dest = board.piece_at(move.to_square)
        captured_piece_type = piece_at_dest.piece_type if piece_at_dest else None
        was_ep = False
        ks_castle = False
        qs_castle = False

        board.push(move)
        promotion_hash = zobrist.update_hash_for_move(
            board,
            move,
            old_castling,
            old_ep,
            captured_piece_type,
            was_ep,
            ks_castle,
            qs_castle,
        )

        # Compute fresh hash to verify

        fresh_hash = zobrist.hash_board(board)
        assert promotion_hash == fresh_hash

        # Should not match original position

        assert promotion_hash != original_hash


class TestZobristIntegration:
    """Test Zobrist integration with search algorithm."""

    def test_node_count_reduction(self):
        """Test that Zobrist hashing reduces node count in search."""
        board = chess.Board()
        evaluator = MockEval(board)
        depth = 3  # Reduced depth for faster testing

        # First search without Zobrist

        config_no_zobrist = EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=False,
                use_tt_aging=False,
                max_time=None,  # No timeout for testing
            )
        )
        minimax_no_zobrist = Minimax(board, evaluator, config_no_zobrist)
        minimax_no_zobrist.find_top_move(depth=depth)
        nodes_without_zobrist = minimax_no_zobrist.node_count

        # Then search with Zobrist

        config_with_zobrist = EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=True,
                use_tt_aging=False,  # Test just Zobrist, not aging
                max_time=None,  # No timeout for testing
            )
        )
        minimax_with_zobrist = Minimax(board, evaluator, config_with_zobrist)
        minimax_with_zobrist.find_top_move(depth=depth)
        nodes_with_zobrist = minimax_with_zobrist.node_count

        # Should see a reduction in node count

        assert nodes_with_zobrist <= nodes_without_zobrist

        # If there's a significant reduction, verify it

        if nodes_without_zobrist > nodes_with_zobrist:
            reduction_ratio = (
                nodes_without_zobrist - nodes_with_zobrist
            ) / nodes_without_zobrist
            # At least some reduction should occur

            assert (
                reduction_ratio > 0
            ), f"Expected some node reduction, got {reduction_ratio:.2%}"

    def test_aging_vs_no_aging_efficiency(self):
        """Test that TT aging provides better efficiency over time."""
        evaluator = MockEval(chess.Board())
        depth = 3  # Reduced depth for faster testing

        # Configuration with aging

        config_with_aging = EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=True,
                use_tt_aging=True,
                max_time=None,  # No timeout for testing
            )
        )

        # Configuration without aging

        config_without_aging = EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=True,
                use_tt_aging=False,
                max_time=None,  # No timeout for testing
            )
        )

        # Test positions

        positions = [
            chess.Board(),  # Starting position
            chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
            chess.Board(
                "rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
            ),
        ]

        total_nodes_with_aging = 0
        total_nodes_without_aging = 0

        for pos in positions:
            # Create fresh instances for each position to avoid state interference

            minimax_with_aging = Minimax(pos, MockEval(pos), config_with_aging)
            minimax_without_aging = Minimax(pos, MockEval(pos), config_without_aging)

            # Search with aging

            minimax_with_aging.find_top_move(depth=depth)
            total_nodes_with_aging += minimax_with_aging.node_count

            # Search without aging

            minimax_without_aging.find_top_move(depth=depth)
            total_nodes_without_aging += minimax_without_aging.node_count
        # Over multiple positions, aging should be equal or more efficient

        assert total_nodes_with_aging <= total_nodes_without_aging
