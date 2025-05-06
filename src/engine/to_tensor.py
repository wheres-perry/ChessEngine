import torch
import chess
import numpy as np
import constants


WHITE_PAWN_PLANE = 0
BLACK_PAWN_PLANE = 1
WHITE_KNIGHT_PLANE = 2
BLACK_KNIGHT_PLANE = 3
WHITE_BISHOP_PLANE = 4
BLACK_BISHOP_PLANE = 5
WHITE_ROOK_PLANE = 6
BLACK_ROOK_PLANE = 7
WHITE_QUEEN_PLANE = 8
BLACK_QUEEN_PLANE = 9
WHITE_KING_PLANE = 10
BLACK_KING_PLANE = 11
TURN_COUNT_PLANE = 12
WHITE_K_CASTLE_PLANE = 13
BLACK_K_CASTLE_PLANE = 14
WHITE_Q_CASTLE_PLANE = 15
BLACK_Q_CASTLE_PLANE = 16
EN_PASSANT_PLANE = 17
MOVE_COUNT_PLANE = 18
DRAW_PLANE_1 = 19
DRAW_PLANE_2 = 20
WHITE_ATTACK_PLANE = 21
BLACK_ATTACK_PLANE = 22
WHITE_PIN_PLANE = 23
BLACK_PIN_PLANE = 24
WHITE_PASSED_PAWN_PLANE = 25
BLACK_PASSED_PAWN_PLANE = 26



def create_piece_plane(board: chess.Board, color: chess.Color, type: chess.PieceType) -> torch.Tensor:
    out = torch.zeros((8, 8), dtype=torch.float32)

    w_pawns = board.pieces(piece_type=chess.PAWN, color=chess.WHITE)
    for sqr in w_pawns:
        r = 7 - (sqr // 8)
        c = sqr % 8
        out[r, c] = 1.0

    return out


def create_tensor(board: chess.Board):
    out: torch.Tensor = torch.zeros((12, 8, 8), dtype=torch.float32)
    
    idx = 0
    for color in (chess.WHITE, chess.BLACK):
        for ptype in constants.EVAL_PIECES:
            out[idx] = create_piece_plane(board, color=color, type=ptype)
            idx += 1

    out[idx] = create_piece_plane(board, color=chess.WHITE, type=chess.KING)
    out[idx + 1] = create_piece_plane(board, color=chess.BLACK, type=chess.KING)
    
