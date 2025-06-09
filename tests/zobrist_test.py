import os
import sys
import time

import chess
import pytest


from src.engine.evaluators.eval import Eval
from src.engine.minimax import Minimax
from src.engine.zobrist import Zobrist


class MockEval(Eval):
    """
    A mock evaluator that returns a predefined score.
    Useful for testing purposes where evaluation complexity is not the focus.
    """

    def __init__(self, board: chess.Board, fixed_score: float):
        super().__init__(board)
        self.fixed_score = fixed_score

    def evaluate(self) -> float:
        return self.fixed_score


def test_zobrist_hash_consistency():
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


def test_zobrist_hash_uniqueness_piece_position():
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


def test_zobrist_hash_uniqueness_turn():
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


def test_zobrist_hash_uniqueness_castling_rights():
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


def test_zobrist_hash_uniqueness_en_passant():
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


STATIC_POSITIONS_FEN = [
    "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    "qk3bNq/N1p1B3/B4Nr1/1p2n2r/2P3PP/3PN3/3N2PR/2RnB2K w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "2kr3r/pb1nqppp/1p2p3/2p1P3/3P2N1/P1PB1N2/5PPP/R2Q1RK1 b - - 0 15",
]


@pytest.mark.parametrize("fen", STATIC_POSITIONS_FEN)
def test_minimax_correctness_with_zobrist(fen):
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
def test_minimax_performance_with_zobrist(fen):
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
