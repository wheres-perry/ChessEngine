"""Piece-Square Tables for handcoded evaluation (Node E1).

Contains middlegame and endgame piece-square tables for all piece types.
Tables are from White's perspective (index 0 = a1, 63 = h8).
"""

from engine._core import moray_core as chess


def make_table(values: list[int]) -> list[float]:
    """Create a 64-square table from 32 values (half board).

    Takes values for ranks 1-4 and mirrors them for ranks 5-8.
    Converts from centipawns to pawns.

    Args:
        values: List of 32 integers (centipawns) for ranks 1-4.

    Returns:
        List of 64 floats (pawns) for all squares.

    """
    as_float = [v / 100.0 for v in values]
    return as_float + as_float[::-1]


# =============================================================================
# Middlegame Piece-Square Tables
# =============================================================================

MG_PAWN_TABLE: list[float] = make_table(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        10,
        10,
        20,
        30,
        30,
        20,
        10,
        10,
        5,
        5,
        10,
        25,
        25,
        10,
        5,
        5,
    ]
)

MG_KNIGHT_TABLE: list[float] = make_table(
    [
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
        -40,
        -20,
        0,
        0,
        0,
        0,
        -20,
        -40,
        -30,
        0,
        10,
        15,
        15,
        10,
        0,
        -30,
        -30,
        5,
        15,
        20,
        20,
        15,
        5,
        -30,
    ]
)

MG_BISHOP_TABLE: list[float] = make_table(
    [
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -10,
        0,
        5,
        10,
        10,
        5,
        0,
        -10,
        -10,
        5,
        5,
        10,
        10,
        5,
        5,
        -10,
    ]
)

MG_ROOK_TABLE: list[float] = make_table(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        5,
        10,
        10,
        10,
        10,
        10,
        10,
        5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
    ]
)

MG_QUEEN_TABLE: list[float] = make_table(
    [
        -20,
        -10,
        -10,
        -5,
        -5,
        -10,
        -10,
        -20,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -10,
        0,
        5,
        5,
        5,
        5,
        0,
        -10,
        -5,
        0,
        5,
        5,
        5,
        5,
        0,
        -5,
    ]
)

MG_KING_TABLE: list[float] = make_table(
    [
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
    ]
)

# =============================================================================
# Endgame Piece-Square Tables
# =============================================================================

EG_PAWN_TABLE: list[float] = make_table(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        80,
        80,
        80,
        80,
        80,
        80,
        80,
        80,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        30,
        30,
        30,
        30,
        30,
        30,
        30,
        30,
    ]
)

EG_KNIGHT_TABLE: list[float] = make_table(
    [
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
        -40,
        -20,
        0,
        0,
        0,
        0,
        -20,
        -40,
        -30,
        0,
        10,
        15,
        15,
        10,
        0,
        -30,
        -30,
        5,
        15,
        20,
        20,
        15,
        5,
        -30,
    ]
)

EG_BISHOP_TABLE: list[float] = make_table(
    [
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -10,
        0,
        5,
        10,
        10,
        5,
        0,
        -10,
        -10,
        5,
        5,
        10,
        10,
        5,
        5,
        -10,
    ]
)

EG_ROOK_TABLE: list[float] = make_table(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        5,
        10,
        10,
        10,
        10,
        10,
        10,
        5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
    ]
)

EG_QUEEN_TABLE: list[float] = make_table(
    [
        -20,
        -10,
        -10,
        -5,
        -5,
        -10,
        -10,
        -20,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -10,
        0,
        5,
        5,
        5,
        5,
        0,
        -10,
        -5,
        0,
        5,
        5,
        5,
        5,
        0,
        -5,
    ]
)

EG_KING_TABLE: list[float] = make_table(
    [
        -50,
        -40,
        -30,
        -20,
        -20,
        -30,
        -40,
        -50,
        -30,
        -20,
        -10,
        0,
        0,
        -10,
        -20,
        -30,
        -30,
        -10,
        20,
        30,
        30,
        20,
        -10,
        -30,
        -30,
        -10,
        30,
        40,
        40,
        30,
        -10,
        -30,
    ]
)

# =============================================================================
# Organized Tables by Piece Type
# =============================================================================

PIECE_SQUARE_TABLES_MG = {
    chess.PAWN: MG_PAWN_TABLE,
    chess.KNIGHT: MG_KNIGHT_TABLE,
    chess.BISHOP: MG_BISHOP_TABLE,
    chess.ROOK: MG_ROOK_TABLE,
    chess.QUEEN: MG_QUEEN_TABLE,
    chess.KING: MG_KING_TABLE,
}

PIECE_SQUARE_TABLES_EG = {
    chess.PAWN: EG_PAWN_TABLE,
    chess.KNIGHT: EG_KNIGHT_TABLE,
    chess.BISHOP: EG_BISHOP_TABLE,
    chess.ROOK: EG_ROOK_TABLE,
    chess.QUEEN: EG_QUEEN_TABLE,
    chess.KING: EG_KING_TABLE,
}
