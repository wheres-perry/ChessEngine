import chess
import torch
from src.engine.constants import *
import logging

logger = logging.getLogger(__name__)


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

DRAW_PLANE_1 = 18
DRAW_PLANE_2 = 19
DRAW_PLANE_3 = 20

WHITE_ATTACK_PLANE = 21
BLACK_ATTACK_PLANE = 22

WHITE_PIN_PLANE = 23
BLACK_PIN_PLANE = 24

WHITE_PASSED_PAWN_PLANE = 25
BLACK_PASSED_PAWN_PLANE = 26

CURRENT_COLOR_PLANE = 27
NUM_PLANES = 28


def create_piece_plane(
    board: chess.Board, color: chess.Color, type: chess.PieceType
) -> torch.Tensor:
    out = torch.zeros((8, 8), dtype=torch.float32)
    pieces = board.pieces(piece_type=type, color=color)
    for sqr in pieces:
        r = 7 - (sqr // 8)
        c = sqr % 8
        out[r, c] = 1.0
    return out


def create_en_passant_plane(board: chess.Board) -> torch.Tensor:
    en_passant_plane = torch.zeros((8, 8), dtype=torch.float32)
    if board.ep_square is not None:
        ep_sq_index = board.ep_square
        row = 7 - (ep_sq_index // 8)
        col = ep_sq_index % 8
        en_passant_plane[row, col] = 1.0
    return en_passant_plane


def create_repetition_planes(
    board: chess.Board,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_2nd_occurrence = torch.zeros((8, 8), dtype=torch.float32)
    is_3rd_or_4th_occurrence = torch.zeros((8, 8), dtype=torch.float32)
    is_5th_or_more_occurrence = torch.zeros((8, 8), dtype=torch.float32)

    if board.is_repetition(2) and not board.is_repetition(3):
        is_2nd_occurrence.fill_(1.0)
        logger.debug("Board %s has 2 repetitions", board)
    if board.is_repetition(3) and not board.is_repetition(5):
        is_3rd_or_4th_occurrence.fill_(1.0)
        logger.debug("Board %s has 3-4 repetitions(Optional Draw)", board)
    if board.is_repetition(5):
        is_5th_or_more_occurrence.fill_(1.0)
        logger.debug("Board %s has 5+ repetitions (Auto draw)", board)
    return is_2nd_occurrence, is_3rd_or_4th_occurrence, is_5th_or_more_occurrence


def create_pinned_pieces_plane(board: chess.Board, color: chess.Color) -> torch.Tensor:
    out = torch.zeros((8, 8), dtype=torch.float32)
    for i in chess.SQUARES:
        piece = board.piece_at(i)
        if piece is not None and piece.color == color:
            if board.is_pinned(color, i):
                r = 7 - (i // 8)
                c = i % 8
                out[r, c] = 1.0
    return out


def create_attacked_squares_plane(
    board: chess.Board, color: chess.Color
) -> torch.Tensor:
    attack_plane = torch.zeros((8, 8), dtype=torch.float32)
    for i in chess.SQUARES:
        if board.is_attacked_by(color, i):
            r = 7 - (i // 8)
            c = i % 8
            attack_plane[r, c] = 1.0
    return attack_plane


def _is_passed_pawn(board: chess.Board, i: chess.Square, color: chess.Color) -> bool:
    pawn_file = chess.square_file(i)
    pawn_rank = chess.square_rank(i)
    opponent_color = not color
    for opp_pawn_i in board.pieces(chess.PAWN, opponent_color):
        opp_pawn_file = chess.square_file(opp_pawn_i)
        opp_pawn_rank = chess.square_rank(opp_pawn_i)
        if abs(pawn_file - opp_pawn_file) <= 1:
            if color == chess.WHITE and opp_pawn_rank > pawn_rank:
                return False
            elif color == chess.BLACK and opp_pawn_rank < pawn_rank:
                return False
    return True


def create_passed_pawns_plane(board: chess.Board, color: chess.Color) -> torch.Tensor:
    out = torch.zeros((8, 8), dtype=torch.float32)
    for i in board.pieces(chess.PAWN, color):
        if _is_passed_pawn(board, i, color):
            r = 7 - (i // 8)
            c = i % 8
            out[r, c] = 1.0
    return out


def create_tensor(board: chess.Board):
    logger.debug("Creating tensor for board: %s", board)
    if not board.is_valid():
        raise ValueError("Invalid board state")
    out: torch.Tensor = torch.zeros((NUM_PLANES, 8, 8), dtype=torch.float32)

    idx: int = 0
    for t in EVAL_PIECES:
        for c in (chess.WHITE, chess.BLACK):
            logger.debug("Adding piece plane for type %s, color %s", t, c)
            out[idx] = create_piece_plane(board, color=c, type=t)
            idx += 1
    # Index 10: White King

    logger.debug("Adding piece plane for white king")
    out[idx] = create_piece_plane(board, color=chess.WHITE, type=chess.KING)
    idx += 1

    # Index 11: Black King

    logger.debug("Adding piece plane for black king")
    out[idx] = create_piece_plane(board, color=chess.BLACK, type=chess.KING)
    idx += 1

    # Index 12: Turn count

    logger.debug("Adding turn count: %d", board.fullmove_number)
    out[idx].fill_(board.fullmove_number / 100)
    idx += 1

    # Index 13: White King-side castle

    logger.debug(
        "Adding white kingside castle rights: %s",
        board.has_kingside_castling_rights(chess.WHITE),
    )
    out[idx].fill_(float(board.has_kingside_castling_rights(chess.WHITE)))
    idx += 1

    # Index 14: Black King-side castle

    logger.debug(
        "Adding black kingside castle rights: %s",
        board.has_kingside_castling_rights(chess.BLACK),
    )
    out[idx].fill_(float(board.has_kingside_castling_rights(chess.BLACK)))
    idx += 1

    # Index 15: White Queen-side castle

    logger.debug(
        "Adding white queenside castle rights: %s",
        board.has_queenside_castling_rights(chess.WHITE),
    )
    out[idx].fill_(float(board.has_queenside_castling_rights(chess.WHITE)))
    idx += 1

    # Index 16: Black Queen-side castle

    logger.debug(
        "Adding black queenside castle rights: %s",
        board.has_queenside_castling_rights(chess.BLACK),
    )
    out[idx].fill_(float(board.has_queenside_castling_rights(chess.BLACK)))
    idx += 1

    # Index 17: En passant
    # TODO: implement en passant

    logger.debug("Adding en passant plane")
    out[idx] = create_en_passant_plane(board)
    idx += 1

    # Index 18-20: Draw planes

    p1, p2, p3 = create_repetition_planes(board)  # Debug log in function
    out[idx] = p1
    idx += 1
    out[idx] = p2
    idx += 1
    out[idx] = p3
    idx += 1

    # Index 21: White attack (Pieces attacked by white)

    logger.debug("Adding white attack plane")
    out[idx] = create_attacked_squares_plane(board, color=chess.WHITE)
    idx += 1

    # Index 22: Black attack (Pieces attacked by black)

    logger.debug("Adding black attack plane")
    out[idx] = create_attacked_squares_plane(board, color=chess.BLACK)
    idx += 1

    # Index 23: White pinned pieces (White pieces pinned by black)

    logger.debug("Adding white pinned pieces plane")
    out[idx] = create_pinned_pieces_plane(board, color=chess.WHITE)
    idx += 1

    # Index 24: Black pinned pieces (Black pieces pinned by white)

    logger.debug("Adding black pinned pieces plane")
    out[idx] = create_pinned_pieces_plane(board, color=chess.BLACK)
    idx += 1

    # Index 25: White passed pawns

    logger.debug("Adding white passed pawns plane")
    out[idx] = create_passed_pawns_plane(board, color=chess.WHITE)
    idx += 1

    # Index 26: Black passed pawns

    logger.debug("Adding black passed pawns plane")
    out[idx] = create_passed_pawns_plane(board, color=chess.BLACK)
    idx += 1

    # Index 27: Current color

    logger.debug("Adding current color plane")
    out[idx].fill_(float(board.turn))  # 0 = Black, 1 = White

    return out
