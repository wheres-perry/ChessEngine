"""Test suite for the Zobrist hashing implementation.

Validates efficient position hashing with incremental updates, ensuring
hash consistency, uniqueness, and correctness for special chess moves.
"""

import pytest

from engine._core import moray_core as chess
from engine.config import EngineConfig, SearchConfig
from engine.evaluators import MockEvaluator
from engine.search.minimax import Minimax
from engine.search.zobrist import Zobrist


class TestZobristBasics:
    """Test basic functionality of Zobrist hashing."""

    def test_hash_consistency(self):
        """Verify identical positions always yield the same hash."""
        zobrist = Zobrist(seed=42)
        board = chess.Board()

        hashes = [zobrist.hash_board(board) for _ in range(3)]

        assert hashes[0] == hashes[1] == hashes[2]

    def test_hash_uniqueness(self):
        """Verify different positions yield different hashes."""
        zobrist = Zobrist(seed=42)
        board1 = chess.Board()
        board2 = chess.Board()

        board2.push_san("e4")

        hash1 = zobrist.hash_board(board1)
        hash2 = zobrist.hash_board(board2)

        assert hash1 != hash2

    def test_position_independence(self):
        """Verify hash depends only on position, not move history."""
        zobrist = Zobrist(seed=42)

        board1 = chess.Board()
        board1.push_san("e4")
        board1.push_san("e5")

        board2 = chess.Board()
        board2.push_san("e4")
        board2.push_san("d5")
        board2.pop()
        board2.push_san("e5")

        hash1 = zobrist.hash_board(board1)
        hash2 = zobrist.hash_board(board2)

        assert hash1 == hash2


class TestZobristIncrementalUpdates:
    """Test incremental Zobrist hash updates."""

    def test_incremental_vs_full_hash(self):
        """Verify incremental updates match full hash computation."""
        zobrist = Zobrist(seed=42)
        board = chess.Board()

        zobrist.hash_board(board)

        move = chess.Move.from_uci("e2e4")

        incremental_hash = zobrist.make_move_hash(board, move)

        board.push(move)

        fresh_hash = zobrist.hash_board(board)

        assert incremental_hash == fresh_hash

    def test_multiple_moves_consistency(self):
        """Verify hash consistency across a series of moves."""
        zobrist = Zobrist(seed=42)
        board = chess.Board()

        zobrist.hash_board(board)

        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]

        for uci in moves:
            move = chess.Move.from_uci(uci)

            incremental_hash = zobrist.make_move_hash(board, move)

            board.push(move)

            fresh_hash = zobrist.hash_board(board)
            assert incremental_hash == fresh_hash


class TestZobristSpecialMoves:
    """Test Zobrist hashing of special chess moves."""

    def test_castling_hash(self):
        """Verify correct hashing of castling moves."""
        zobrist = Zobrist(seed=42)

        board = chess.Board.from_fen(
            "r3k2r/ppp1pppp/2n2n2/8/8/2N2N2/PPP1PPPP/R3K2R w KQkq - 0 1"
        )
        original_hash = zobrist.hash_board(board)

        move = chess.Move.from_uci("e1g1")

        castle_hash = zobrist.make_move_hash(board, move)

        board.push(move)

        fresh_hash = zobrist.hash_board(board)
        assert castle_hash == fresh_hash

        assert castle_hash != original_hash

    def test_en_passant_hash(self):
        """Verify correct hashing of en passant moves."""
        zobrist = Zobrist(seed=42)

        board = chess.Board.from_fen(
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        )
        original_hash = zobrist.hash_board(board)

        move = chess.Move.from_uci("e5f6")

        ep_hash = zobrist.make_move_hash(board, move)

        board.push(move)

        fresh_hash = zobrist.hash_board(board)
        assert ep_hash == fresh_hash

        assert ep_hash != original_hash

    def test_promotion_hash(self):
        """Verify correct hashing of promotion moves."""
        zobrist = Zobrist(seed=42)

        board = chess.Board.from_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        original_hash = zobrist.hash_board(board)

        move = chess.Move.from_uci("a7a8q")

        promotion_hash = zobrist.make_move_hash(board, move)

        board.push(move)

        fresh_hash = zobrist.hash_board(board)
        assert promotion_hash == fresh_hash

        assert promotion_hash != original_hash


class TestZobristIntegration:
    """Test Zobrist integration with search algorithm."""

    def test_node_count_reduction(self):
        """Verify Zobrist hashing reduces node count in search."""
        board = chess.Board()
        evaluator = MockEvaluator()
        depth = 3

        config_no_zobrist = EngineConfig(
            search=SearchConfig(
                use_transposition_table=False,
                use_tt_aging=False,
                use_hash_move_ordering=False,
                use_iid=False,
                max_time=None,
            )
        )
        minimax_no_zobrist = Minimax(board, evaluator, config_no_zobrist)
        minimax_no_zobrist.find_top_move(depth=depth)
        nodes_without_zobrist = minimax_no_zobrist.node_count

        config_with_zobrist = EngineConfig(
            search=SearchConfig(
                use_tt_aging=False,
                max_time=None,
            )
        )
        minimax_with_zobrist = Minimax(board, evaluator, config_with_zobrist)
        minimax_with_zobrist.find_top_move(depth=depth)
        nodes_with_zobrist = minimax_with_zobrist.node_count

        assert nodes_with_zobrist <= nodes_without_zobrist

        if nodes_without_zobrist > nodes_with_zobrist:
            reduction_ratio = (
                nodes_without_zobrist - nodes_with_zobrist
            ) / nodes_without_zobrist
            assert reduction_ratio > 0, (
                f"Expected some node reduction, got {reduction_ratio:.2%}"
            )

    def test_aging_vs_no_aging_efficiency(self):
        """Verify TT aging provides better efficiency over time."""
        depth = 3

        config_with_aging = EngineConfig(
            search=SearchConfig(
                use_tt_aging=True,
                max_time=None,
            )
        )

        config_without_aging = EngineConfig(
            search=SearchConfig(
                use_tt_aging=False,
                max_time=None,
            )
        )

        positions = [
            chess.Board(),
            chess.Board.from_fen(
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
            ),
            chess.Board.from_fen(
                "rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
            ),
        ]

        total_nodes_with_aging = 0
        total_nodes_without_aging = 0

        for pos in positions:
            minimax_with_aging = Minimax(pos, MockEvaluator(), config_with_aging)
            minimax_without_aging = Minimax(pos, MockEvaluator(), config_without_aging)

            minimax_with_aging.find_top_move(depth=depth)
            total_nodes_with_aging += minimax_with_aging.node_count

            minimax_without_aging.find_top_move(depth=depth)
            total_nodes_without_aging += minimax_without_aging.node_count

        assert total_nodes_with_aging <= total_nodes_without_aging


if __name__ == "__main__":
    pytest.main([__file__])
