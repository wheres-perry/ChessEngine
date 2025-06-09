import time
from typing import Optional

import chess
import pytest

from src.engine.constants import PIECE_VALUES
from src.engine.evaluators.eval import Eval
from src.engine.minimax import Minimax
from src.engine.zobrist import Zobrist


class MockEval(Eval):
    """
    A mock evaluator that returns a predefined score.
    Useful for testing purposes.
    """

    def __init__(self, board: chess.Board, fixed_score: float):
        super().__init__(board)
        self.fixed_score = fixed_score
        self.score = fixed_score

    def evaluate(self) -> float:
        """
        Returns the predefined fixed score.
        """
        self.score = self.fixed_score
        return self.score

    def set_score(self, score: float):
        """
        Allows updating the fixed score after instantiation.
        """
        self.fixed_score = score
        self.score = score


class TestMoveOrdering:
    """Tests for the move ordering functionality in Minimax."""

    def test_order_moves_returns_list(self):
        """Test that order_moves returns a list of moves."""
        board = chess.Board()
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)

        moves = list(board.legal_moves)
        ordered_moves = minimax.order_moves(moves)

        assert isinstance(ordered_moves, list)
        assert len(ordered_moves) == len(moves)

    def test_order_moves_with_empty_list(self):
        """Test that order_moves handles empty move list."""
        board = chess.Board()
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)

        empty_moves = []
        ordered_moves = minimax.order_moves(empty_moves)

        assert ordered_moves == []

    def test_order_moves_with_single_move(self):
        """Test that order_moves handles a single move."""
        board = chess.Board("8/8/8/r7/6PK/8/6k1/8 w - - 0 1")  # Only pawn can move
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)

        moves = list(board.legal_moves)
        ordered_moves = minimax.order_moves(moves)

        assert len(ordered_moves) == len(moves)
        assert ordered_moves[0] in moves

    def test_order_moves_with_two_moves(self):
        """Test a pawn move vs capture move."""
        board = chess.Board("8/8/8/r4q2/6PK/8/6k1/8 w - - 0 1")
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)

        expected_ordered_moves = [
            chess.Move.from_uci("g4f5"),  # Capture move (should be prioritized)
            chess.Move.from_uci("g4g5"),  # Quiet move
        ]

        moves = list(board.legal_moves)
        ordered_moves = minimax.order_moves(moves)

        # Assert that the ordered moves match our expectations
        assert ordered_moves == expected_ordered_moves
        assert len(ordered_moves) == 2

    def test_order_moves_with_five_moves(self):
        """Test ordering with multiple captures of different values."""
        board = chess.Board("8/8/8/r4q2/6PK/1r6/2P5/7k w - - 0 1")

        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)

        # Get all legal moves and verify we have the expected number
        all_legal_moves = list(board.legal_moves)
        assert (
            len(all_legal_moves) == 5
        ), f"Expected 5 legal moves, got {len(all_legal_moves)}: {[move.uci() for move in all_legal_moves]}"

        # Order the moves
        ordered_moves = minimax.order_moves(all_legal_moves)
        assert (
            len(ordered_moves) == 5
        ), f"Expected 5 ordered moves, got {len(ordered_moves)}"

        # Define the expected high-priority captures
        queen_capture = chess.Move.from_uci("g4f5")  # Pawn captures queen
        rook_capture = chess.Move.from_uci("c2b3")  # Pawn captures rook

        # Verify these moves are actually legal in the position
        assert (
            queen_capture in all_legal_moves
        ), f"Queen capture {queen_capture.uci()} not in legal moves: {[m.uci() for m in all_legal_moves]}"
        assert (
            rook_capture in all_legal_moves
        ), f"Rook capture {rook_capture.uci()} not in legal moves: {[m.uci() for m in all_legal_moves]}"

        # The first two moves should be the captures, prioritized by piece value (queen > rook)
        assert (
            ordered_moves[0] == queen_capture
        ), f"First move should be queen capture, got {ordered_moves[0].uci()}"
        assert (
            ordered_moves[1] == rook_capture
        ), f"Second move should be rook capture, got {ordered_moves[1].uci()}"

        # The remaining moves should be the quiet moves (order doesn't matter)
        capture_moves = {queen_capture, rook_capture}
        expected_quiet_moves = set(all_legal_moves) - capture_moves
        actual_quiet_moves = set(ordered_moves[2:])

        assert (
            actual_quiet_moves == expected_quiet_moves
        ), f"Quiet moves don't match. Expected: {[m.uci() for m in expected_quiet_moves]}, Got: {[m.uci() for m in actual_quiet_moves]}"

    def test_order_moves_with_check_moves(self):
        """Test that order_moves prioritizes check moves."""
        board = chess.Board("2r5/8/8/5k2/7b/3PP3/7r/3K4 w - - 0 1")
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)

        expected_ordered_moves = [
            chess.Move.from_uci("e3e4"),  # Check move
            chess.Move.from_uci("d3d4"),  # quiet move
        ]

        moves = list(board.legal_moves)
        ordered_moves = minimax.order_moves(moves)

        # Assert that the ordered moves match our expectations
        assert ordered_moves == expected_ordered_moves

    def test_order_moves_with_stalemate_moves(self):
        """Test that order_moves handles stalemate moves."""
        board = chess.Board(
            "8/8/8/2R5/2p2kN1/2N4P/PP3PP1/6K1 b - - 0 33"
        )  # Black to move, stalemate
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)

        moves = list(board.legal_moves)
        ordered_moves = minimax.order_moves(moves)

        assert len(ordered_moves) == 0, "In stalemate, there should be no legal moves."

    def test_order_moves_with_player_alr_checkmated(self):
        """Test that order_moves handles checkmate positions."""
        board = chess.Board(
            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        )  # Black to move, checkmated
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)

        moves = list(board.legal_moves)
        ordered_moves = minimax.order_moves(moves)

        assert len(ordered_moves) == 0, "In checkmate, there should be no legal moves."

    def test_complex_move_ordering(self):
        """Test complex move ordering with captures, checks, and quiet moves."""
        board = chess.Board("1k6/8/8/rpp1bp2/1P3P2/3K4/5q2/N6N w - - 0 1")
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)

        # Get all legal moves and verify we have the expected number
        all_legal_moves = list(board.legal_moves)
        expected_total_moves = 7
        assert (
            len(all_legal_moves) == expected_total_moves
        ), f"Expected {expected_total_moves} legal moves, got {len(all_legal_moves)}: {[move.uci() for move in all_legal_moves]}"

        # Order the moves
        ordered_moves = minimax.order_moves(all_legal_moves)
        assert (
            len(ordered_moves) == expected_total_moves
        ), f"Expected {expected_total_moves} ordered moves, got {len(ordered_moves)}"

        # Define the expected high-priority captures (in order of piece value) - UCI strings only
        expected_priority_capture_ucis = [
            "h1f2",  # Knight captures queen (highest value)
            "b4a5",  # Pawn captures rook
            "f4e5",  # Pawn captures bishop
            "b4c5",  # Pawn captures pawn (lowest value capture)
        ]

        # Convert to UCI strings for comparison
        all_legal_move_ucis = [move.uci() for move in all_legal_moves]
        ordered_move_ucis = [move.uci() for move in ordered_moves]

        # Verify these moves are actually legal in the position
        for i, move_uci in enumerate(expected_priority_capture_ucis):
            assert (
                move_uci in all_legal_move_ucis
            ), f"Priority capture {i+1} ({move_uci}) not in legal moves: {all_legal_move_ucis}"
        
        # The first 4 moves should be the captures in exact order (by piece value)
        for i, expected_uci in enumerate(expected_priority_capture_ucis):
            assert (
                ordered_move_ucis[i] == expected_uci
            ), f"Move {i+1} should be {expected_uci}, got {ordered_move_ucis[i]}"
        
        # Define the expected quiet moves UCI strings (order doesn't matter for these)
        expected_quiet_move_ucis = {
            "a1b3",  # Knight move
            "a1c2",  # Knight move
            "h1g3",  # Knight move
        }

        # Verify quiet moves are legal
        for move_uci in expected_quiet_move_ucis:
            assert (
                move_uci in all_legal_move_ucis
            ), f"Quiet move {move_uci} not in legal moves: {all_legal_move_ucis}"
        
        # The remaining moves should be the quiet moves (order doesn't matter)
        calculated_quiet_move_ucis = set(all_legal_move_ucis) - set(
            expected_priority_capture_ucis
        )
        actual_quiet_move_ucis = set(ordered_move_ucis[4:])

        # Verify we have the expected quiet moves
        assert (
            actual_quiet_move_ucis == expected_quiet_move_ucis
        ), f"Expected quiet moves: {sorted(expected_quiet_move_ucis)}, Got: {sorted(actual_quiet_move_ucis)}"
        assert (
            calculated_quiet_move_ucis == expected_quiet_move_ucis
        ), f"Calculated quiet moves don't match expected. Calculated: {sorted(calculated_quiet_move_ucis)}, Expected: {sorted(expected_quiet_move_ucis)}"


class TestZobristHashing:
    """Tests for Zobrist hashing functionality."""

    def test_zobrist_hash_consistency(self):
        """Test that the same board position produces the same hash from the same Zobrist instance."""
        zobrist = Zobrist()
        board1 = chess.Board()
        board2 = chess.Board()

        hash1 = zobrist.hash_board(board1)
        hash2 = zobrist.hash_board(board2)
        assert hash1 == hash2, "Hashes for identical initial positions should be the same."

        move = chess.Move.from_uci("e2e4")
        board1.push(move)
        hash_after_move = zobrist.hash_board(board1)
        assert hash1 != hash_after_move, "Hash should change after a move."
        board1.pop()
        hash_after_pop = zobrist.hash_board(board1)
        assert hash1 == hash_after_pop, "Hash should revert after unmaking the move."

    def test_zobrist_hash_uniqueness_piece_position(self):
        """Test that different piece positions produce different hashes."""
        zobrist = Zobrist()
        board1 = chess.Board()
        hash1 = zobrist.hash_board(board1)

        board2 = board1.copy()
        board2.push(chess.Move.from_uci("g1f3"))
        hash2 = zobrist.hash_board(board2)
        assert (
            hash1 != hash2
        ), "Different piece positions should result in different hashes."

    def test_zobrist_hash_uniqueness_turn(self):
        """Test that different turns produce different hashes."""
        zobrist = Zobrist()
        board1 = chess.Board()
        hash1 = zobrist.hash_board(board1)

        board2 = board1.copy()
        board_white_turn = chess.Board()
        hash_white_turn = zobrist.hash_board(board_white_turn)

        board_black_turn = chess.Board()
        board_black_turn.turn = chess.BLACK
        hash_black_turn = zobrist.hash_board(board_black_turn)

        assert (
            hash_white_turn != hash_black_turn
        ), "Different turns should result in different hashes."

    def test_zobrist_hash_uniqueness_castling_rights(self):
        """Test that different castling rights produce different hashes."""
        zobrist = Zobrist()
        board1 = chess.Board()
        hash1 = zobrist.hash_board(board1)

        board2 = board1.copy()
        board2.castling_rights &= ~chess.BB_H1
        hash2 = zobrist.hash_board(board2)
        assert (
            hash1 != hash2
        ), "Hash should change with different white kingside castling rights."

        board3 = board1.copy()
        board3.castling_rights &= ~chess.BB_A8
        hash3 = zobrist.hash_board(board3)
        assert (
            hash1 != hash3
        ), "Hash should change with different black queenside castling rights."
        assert (
            hash2 != hash3
        ), "Different specific castling right changes should also differ."

    def test_zobrist_hash_uniqueness_en_passant(self):
        """Test that different en passant squares produce different hashes."""
        zobrist = Zobrist()

        fen_with_ep = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq c6 0 2"
        board_with_ep = chess.Board(fen_with_ep)
        assert board_with_ep.ep_square == chess.C6
        hash_with_ep = zobrist.hash_board(board_with_ep)

        fen_without_ep_same_pieces = (
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
        )
        board_without_ep = chess.Board(fen_without_ep_same_pieces)
        assert board_without_ep.ep_square is None
        hash_without_ep = zobrist.hash_board(board_without_ep)

        assert (
            hash_with_ep != hash_without_ep
        ), "Hash should change with different en passant states."

        fen_different_ep = "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 2"
        board_different_ep = chess.Board(fen_different_ep)
        assert board_different_ep.ep_square == chess.E6
        hash_different_ep = zobrist.hash_board(board_different_ep)
        assert (
            hash_with_ep != hash_different_ep
        ), "Hash should change with different en passant squares."


class TestMinimaxWithZobrist:
    """Tests for Minimax algorithm with Zobrist hashing."""

    STATIC_POSITIONS_FEN = [
        "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
        "qk3bNq/N1p1B3/B4Nr1/1p2n2r/2P3PP/3PN3/3N2PR/2RnB2K w - - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "2kr3r/pb1nqppp/1p2p3/2p1P3/3P2N1/P1PB1N2/5PPP/R2Q1RK1 b - - 0 15",
    ]

    @pytest.mark.parametrize("fen", STATIC_POSITIONS_FEN)
    def test_minimax_correctness_with_zobrist(self, fen):
        """
        Tests that Minimax with Zobrist hashing produces the same result (score and move)
        as without Zobrist hashing.
        """
        board = chess.Board(fen)
        evaluator = MockEval(board, 0.0)

        minimax_no_zobrist = Minimax(board.copy(), evaluator, use_zobrist=False)
        score_no_zobrist, move_no_zobrist = minimax_no_zobrist.find_top_move(depth=6)

        minimax_with_zobrist = Minimax(board.copy(), evaluator, use_zobrist=True)
        score_with_zobrist, move_with_zobrist = minimax_with_zobrist.find_top_move(depth=6)

        assert (
            score_no_zobrist == score_with_zobrist
        ), "Scores should be identical with or without Zobrist."
        assert (
            move_no_zobrist == move_with_zobrist
        ), "Best moves should be identical with or without Zobrist."

    @pytest.mark.parametrize("fen", STATIC_POSITIONS_FEN)
    def test_minimax_performance_with_zobrist(self, fen):
        """
        Tests that Minimax with Zobrist hashing is faster than without Zobrist hashing.
        """
        board = chess.Board(fen)
        evaluator = MockEval(board, 0.0)

        minimax_no_zobrist = Minimax(board.copy(), evaluator, use_zobrist=False)
        start_time_no_zobrist = time.perf_counter()
        score_no_zobrist, move_no_zobrist = minimax_no_zobrist.find_top_move(depth=6)
        end_time_no_zobrist = time.perf_counter()
        duration_no_zobrist = end_time_no_zobrist - start_time_no_zobrist
        nodes_no_zobrist = minimax_no_zobrist.node_count

        minimax_with_zobrist = Minimax(board.copy(), evaluator, use_zobrist=True)
        start_time_with_zobrist = time.perf_counter()
        score_with_zobrist, move_with_zobrist = minimax_with_zobrist.find_top_move(depth=6)
        end_time_with_zobrist = time.perf_counter()
        duration_with_zobrist = end_time_with_zobrist - start_time_with_zobrist
        nodes_with_zobrist = minimax_with_zobrist.node_count

        print(f"\nFEN: {fen} (Depth: 6)")
        print(
            f"  No Zobrist: Score={score_no_zobrist}, Move={move_no_zobrist}, Nodes={nodes_no_zobrist}, Time={duration_no_zobrist:.4f}s"
        )
        print(
            f"  With Zobrist: Score={score_with_zobrist}, Move={move_with_zobrist}, Nodes={nodes_with_zobrist}, Time={duration_with_zobrist:.4f}s"
        )

        assert (
            nodes_with_zobrist <= nodes_no_zobrist
        ), f"Zobrist should lead to fewer or equal nodes. No_Z_nodes: {nodes_no_zobrist}, Z_nodes: {nodes_with_zobrist}"

        assert duration_no_zobrist > 0, "Duration without Zobrist should be greater than 0"
        assert duration_with_zobrist > 0, "Duration with Zobrist should be greater than 0"

        assert (
            duration_with_zobrist <= duration_no_zobrist * 0.95
        ), f"Zobrist version should be at least 5% faster. No_Z_time: {duration_no_zobrist:.4f}, Z_time: {duration_with_zobrist:.4f}"

        speedup_factor = duration_no_zobrist / duration_with_zobrist
        print(
            f"  Speedup factor (No_Zobrist_Time / With_Zobrist_Time): {speedup_factor:.2f}x"
        )