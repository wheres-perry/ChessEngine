"""Helper functions for tests to reduce code duplication."""

from engine._core import chess_engine_core as chess


def make_candidate_move(board: chess.Board, candidate_moves: list[str]) -> bool:
    """
    Try to make one of the candidate moves on the board.

    Args:
        board: The chess board to make a move on
        candidate_moves: List of candidate moves in SAN notation

    Returns:
        True if a move was made, False otherwise
    """
    for candidate_move in candidate_moves:
        try:
            board.push_san(candidate_move)
            return True
        except RuntimeError:
            continue
    return False


def make_any_legal_move(board: chess.Board) -> bool:
    """
    Make any legal move on the board.

    Args:
        board: The chess board to make a move on

    Returns:
        True if a move was made, False if no legal moves available
    """
    legal_moves = list(board.legal_moves)
    if legal_moves:
        board.push(legal_moves[0])
        return True
    return False


def make_test_move(board: chess.Board) -> bool:
    """
    Make a test move on the board, trying common moves first.

    Args:
        board: The chess board to make a move on

    Returns:
        True if a move was made, False if no moves available
    """
    if board.turn == chess.WHITE:
        white_candidates = ["Bc4", "d4", "Nc3", "Nf3", "Qe2", "0-0", "h3"]
        if make_candidate_move(board, white_candidates):
            return True
    else:
        black_candidates = ["Nf6", "d5", "e6", "Bc5", "0-0", "h6"]
        if make_candidate_move(board, black_candidates):
            return True

    return make_any_legal_move(board)
