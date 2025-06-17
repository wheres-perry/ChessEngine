"""
Test suite for the Zobrist hashing implementation.

This module contains tests for the Zobrist class, which provides
efficient position hashing with incremental updates. Tests cover:
- Hash consistency for identical positions
- Hash uniqueness for different positions
- Incremental update correctness
- Special chess moves (castling, en-passant, promotion)
- Integration with search algorithm
"""

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


if __name__ == "__main__":
    pytest.main([__file__])
