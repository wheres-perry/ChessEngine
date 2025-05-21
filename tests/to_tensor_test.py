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


def print_tensor(tensor):
    """Prints a 2D tensor in full, one row per line."""
    for row in tensor:
        print(" ".join([str(x.item()) for x in row]))


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
    assert torch.equal(expected_tensor[TURN_COUNT_PLANE], torch.tensor(
        INIT_TURN, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_K_CASTLE_PLANE], torch.tensor(
        INIT_W_K_CASTLE, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_K_CASTLE_PLANE], torch.tensor(
        INIT_B_K_CASTLE, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_Q_CASTLE_PLANE], torch.tensor(
        INIT_W_Q_CASTLE, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_Q_CASTLE_PLANE], torch.tensor(
        INIT_B_Q_CASTLE, dtype=torch.float32))
    assert torch.equal(expected_tensor[EN_PASSANT_PLANE], torch.tensor(
        INIT_EN_PASSANT, dtype=torch.float32))
    assert torch.equal(expected_tensor[DRAW_PLANE_1], torch.tensor(
        INIT_R1, dtype=torch.float32))
    assert torch.equal(expected_tensor[DRAW_PLANE_2], torch.tensor(
        INIT_R2, dtype=torch.float32))
    assert torch.equal(expected_tensor[DRAW_PLANE_3], torch.tensor(
        INIT_R3, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_ATTACK_PLANE], torch.tensor(
        INIT_W_ATTACK, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_ATTACK_PLANE], torch.tensor(
        INIT_B_ATTACK, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_PIN_PLANE], torch.tensor(
        INIT_W_PIN, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_PIN_PLANE], torch.tensor(
        INIT_B_PIN, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_PASSED_PAWN_PLANE], torch.tensor(
        INIT_W_PP, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_PASSED_PAWN_PLANE], torch.tensor(
        INIT_B_PP, dtype=torch.float32))
    assert torch.equal(expected_tensor[CURRENT_COLOR_PLANE], torch.tensor(
        INIT_CUR_PLAYER, dtype=torch.float32))


def test_castle_pin_ep():
    # A board with an en passant available, a pin present, and modified castling rights
    # Solution tensors created by hand
    board = board_from_pgn_string(
        """1. e4 c6 2. d4 d5 3. Nc3 Nf6 4. exd5 Bg4
        5. Nf3 Bxf3 6. Qxf3 e6 7. dxe6 Nd5
        8. Nxd5 cxd5 9. Bb5+ Nc6 10. O-O Rc8
        11. Bd2 f5 12. Qh3 a5 13. g3 a4 14. b4"""
    )
    expected_tensor = create_tensor(board)

    W_P = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]

    W_B = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    W_N = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    W_R = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0],
    ]
    W_Q = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    W_K = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ]
    B_P = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    B_B = [
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    B_N = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    B_R = [
        [0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    B_Q = [
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    B_K = [
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]

    TURN = [
        [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
        [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
        [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
        [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
        [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
        [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
        [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
        [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
    ]

    W_K_CASTLE = [[0]*8 for _ in range(8)]
    W_Q_CASTLE = [[0]*8 for _ in range(8)]
    B_K_CASTLE = [[1]*8 for _ in range(8)]
    B_Q_CASTLE = [[0]*8 for _ in range(8)]

    EN_PASSANT = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]]

    R1 = [[0]*8 for _ in range(8)]
    R2 = [[0]*8 for _ in range(8)]
    R3 = [[0]*8 for _ in range(8)]

    W_ATTACK = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1],
        [0, 1, 1, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]]
    B_ATTACK = [
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]]

    W_PIN = [[0]*8 for _ in range(8)]

    B_PIN = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]

    W_PP = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    B_PP = [[0]*8 for _ in range(8)]

    CUR_PLAYER = [[0]*8 for _ in range(8)]

    assert torch.equal(expected_tensor[WHITE_PAWN_PLANE], torch.tensor(
        W_P, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_PAWN_PLANE], torch.tensor(
        B_P, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_KNIGHT_PLANE], torch.tensor(
        W_N, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_KNIGHT_PLANE], torch.tensor(
        B_N, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_BISHOP_PLANE], torch.tensor(
        W_B, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_BISHOP_PLANE], torch.tensor(
        B_B, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_ROOK_PLANE], torch.tensor(
        W_R, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_ROOK_PLANE], torch.tensor(
        B_R, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_QUEEN_PLANE], torch.tensor(
        W_Q, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_QUEEN_PLANE], torch.tensor(
        B_Q, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_KING_PLANE], torch.tensor(
        W_K, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_KING_PLANE], torch.tensor(
        B_K, dtype=torch.float32))
    assert torch.equal(expected_tensor[TURN_COUNT_PLANE], torch.tensor(
        TURN, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_K_CASTLE_PLANE], torch.tensor(
        W_K_CASTLE, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_K_CASTLE_PLANE], torch.tensor(
        B_K_CASTLE, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_Q_CASTLE_PLANE], torch.tensor(
        W_Q_CASTLE, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_Q_CASTLE_PLANE], torch.tensor(
        B_Q_CASTLE, dtype=torch.float32))
    assert torch.equal(expected_tensor[EN_PASSANT_PLANE], torch.tensor(
        EN_PASSANT, dtype=torch.float32))
    assert torch.equal(expected_tensor[DRAW_PLANE_1], torch.tensor(
        R1, dtype=torch.float32))
    assert torch.equal(expected_tensor[DRAW_PLANE_2], torch.tensor(
        R2, dtype=torch.float32))
    assert torch.equal(expected_tensor[DRAW_PLANE_3], torch.tensor(
        R3, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_ATTACK_PLANE], torch.tensor(
        W_ATTACK, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_ATTACK_PLANE], torch.tensor(
        B_ATTACK, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_PIN_PLANE], torch.tensor(
        W_PIN, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_PIN_PLANE], torch.tensor(
        B_PIN, dtype=torch.float32))
    assert torch.equal(expected_tensor[WHITE_PASSED_PAWN_PLANE], torch.tensor(
        W_PP, dtype=torch.float32))
    assert torch.equal(expected_tensor[BLACK_PASSED_PAWN_PLANE], torch.tensor(
        B_PP, dtype=torch.float32))
    assert torch.equal(expected_tensor[CURRENT_COLOR_PLANE], torch.tensor(
        CUR_PLAYER, dtype=torch.float32))


if __name__ == '__main__':
    print("Run tests using 'pytest your_test_file_name.py'")