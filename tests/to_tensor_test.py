from multiprocessing.pool import INIT
import pytest
import torch
import chess
from chess import pgn
import io
import pytest
import torch
import chess
from src.engine.to_tensor import *


INIT_W_P = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
INIT_W_B = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
]
INIT_W_N = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
]
INIT_W_R = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1],
]
INIT_W_Q = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
]
INIT_W_K = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
]
INIT_B_P = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
INIT_B_B = [
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
INIT_B_N = [
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
INIT_B_R = [
    [1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
INIT_B_Q = [
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
INIT_B_K = [
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]

INIT_TURN = [
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
]

INIT_W_K_CASTLE = [[1]*8 for _ in range(8)]
INIT_W_Q_CASTLE = [[1]*8 for _ in range(8)]
INIT_B_K_CASTLE = [[1]*8 for _ in range(8)]
INIT_B_Q_CASTLE = [[1]*8 for _ in range(8)]

INIT_EN_PASSANT = [[0]*8 for _ in range(8)]

INIT_R1 = [[0]*8 for _ in range(8)]
INIT_R2 = [[0]*8 for _ in range(8)]
INIT_R3 = [[0]*8 for _ in range(8)]

INIT_W_ATTACK = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0]
]
INIT_B_ATTACK = [
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
]


INIT_W_PIN = [[0]*8 for _ in range(8)]
INIT_B_PIN = [[0]*8 for _ in range(8)]

INIT_W_PP = [[0]*8 for _ in range(8)]
INIT_B_PP = [[0]*8 for _ in range(8)]

INIT_CUR_PLAYER = [[1]*8 for _ in range(8)]

# Helper Functions


def board_from_pgn_string(pgn_string: str) -> chess.Board:
    """Helper to get a board from a PGN string after all moves."""
    pgn_str = io.StringIO(pgn_string)
    game = pgn.read_game(pgn_str)
    if game is None:
        raise ValueError("Could not parse PGN string")
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board


def board_from_fen(fen_string: str) -> chess.Board:
    """Helper to get a board from a FEN string."""
    return chess.Board(fen_string)

# ===========================================================


def test_initial_position_tensor():

    board = chess.Board()
    expected_tensor = create_tensor(board)

    assert torch.equal(expected_tensor[WHITE_PAWN_PLANE], torch.tensor(
        INIT_W_P, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_PAWN_PLANE], torch.tensor(
        INIT_B_P, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_KNIGHT_PLANE], torch.tensor(
        INIT_W_N, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_KNIGHT_PLANE], torch.tensor(
        INIT_B_N, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_BISHOP_PLANE], torch.tensor(
        INIT_W_B, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_BISHOP_PLANE], torch.tensor(
        INIT_B_B, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_ROOK_PLANE], torch.tensor(
        INIT_W_R, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_ROOK_PLANE], torch.tensor(
        INIT_B_R, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_QUEEN_PLANE], torch.tensor(
        INIT_W_Q, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_QUEEN_PLANE], torch.tensor(
        INIT_B_Q, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_KING_PLANE], torch.tensor(
        INIT_W_K, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_KING_PLANE], torch.tensor(
        INIT_B_K, dtype=torch.float32))
    # Index 12: Turn count
    assert torch.equal(expected_tensor[TURN_COUNT_PLANE], torch.tensor(
        INIT_TURN, dtype=torch.float32))
    # Index 13: White King-side castle
    assert torch.equal(expected_tensor[WHITE_K_CASTLE_PLANE], torch.tensor(
        INIT_W_K_CASTLE, dtype=torch.float32))
    # Index 14: Black King-side castle
    assert torch.equal(expected_tensor[BLACK_K_CASTLE_PLANE], torch.tensor(
        INIT_B_K_CASTLE, dtype=torch.float32))
    # Index 15: White Queen-side castle
    assert torch.equal(expected_tensor[WHITE_Q_CASTLE_PLANE], torch.tensor(
        INIT_W_Q_CASTLE, dtype=torch.float32))
    # Index 16: Black Queen-side castle
    assert torch.equal(expected_tensor[BLACK_Q_CASTLE_PLANE], torch.tensor(
        INIT_B_Q_CASTLE, dtype=torch.float32))
    # Index 17: En passant
    assert torch.equal(expected_tensor[EN_PASSANT_PLANE], torch.tensor(
        INIT_EN_PASSANT, dtype=torch.float32))
    # Index 18-20: Draw planes
    assert torch.equal(expected_tensor[DRAW_PLANE_1], torch.tensor(
        INIT_R1, dtype=torch.float32))
    assert torch.equal(expected_tensor[DRAW_PLANE_2], torch.tensor(
        INIT_R2, dtype=torch.float32))
    assert torch.equal(expected_tensor[DRAW_PLANE_3], torch.tensor(
        INIT_R3, dtype=torch.float32))
    # Index 21: White attack (Pieces attacked by white)
    assert torch.equal(expected_tensor[WHITE_ATTACK_PLANE], torch.tensor(
        INIT_W_ATTACK, dtype=torch.float32))
    # Index 22: Black attack (Pieces attacked by black)
    assert torch.equal(expected_tensor[BLACK_ATTACK_PLANE], torch.tensor(
        INIT_B_ATTACK, dtype=torch.float32))
    # Index 23: White pinned pieces (White pieces pinned by black)
    assert torch.equal(expected_tensor[WHITE_PIN_PLANE], torch.tensor(
        INIT_W_PIN, dtype=torch.float32))
    # Index 24: Black pinned pieces (Black pieces pinned by white)
    assert torch.equal(expected_tensor[BLACK_PIN_PLANE], torch.tensor(
        INIT_B_PIN, dtype=torch.float32))
    # Index 25: White passed pawns
    assert torch.equal(expected_tensor[WHITE_PASSED_PAWN_PLANE], torch.tensor(
        INIT_W_PP, dtype=torch.float32))
    # Index 26: Black passed pawns
    assert torch.equal(expected_tensor[BLACK_PASSED_PAWN_PLANE], torch.tensor(
        INIT_B_PP, dtype=torch.float32))
    # Index 27: Current color
    assert torch.equal(expected_tensor[CURRENT_COLOR_PLANE], torch.tensor(
        INIT_CUR_PLAYER, dtype=torch.float32))


if __name__ == '__main__':
    print("Run tests using 'pytest your_test_file_name.py'")
