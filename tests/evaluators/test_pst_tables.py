"""Tests for Piece-Square Tables.

Validate table structure, value ranges, and symmetry properties
to ensure PST-based evaluation behaves correctly across all phases.
"""

import chess as pychess
import pytest

from engine._core import chess_engine_core as chess
from engine.evaluators.pst_tables import (
    PIECE_SQUARE_TABLES_EG,
    PIECE_SQUARE_TABLES_MG,
    make_table,
)

# Square constants from python-chess (integers 0-63, a1=0 ... h8=63)
# Needed because the C++ module doesn't export named square constants.
A1, B1, C1, D1, E1, F1, G1, H1 = range(8)
A2, B2, C2, D2, E2, F2, G2, H2 = range(8, 16)
A3, B3, C3, D3, E3, F3, G3, H3 = range(16, 24)
A4, B4, C4, D4, E4, F4, G4, H4 = range(24, 32)
A5, B5, C5, D5, E5, F5, G5, H5 = range(32, 40)
A6, B6, C6, D6, E6, F6, G6, H6 = range(40, 48)
A7, B7, C7, D7, E7, F7, G7, H7 = range(48, 56)
A8, B8, C8, D8, E8, F8, G8, H8 = range(56, 64)


class TestPSTStructure:
    """Test structure and format of PST tables."""

    def test_mg_tables_exist_for_all_pieces(self):
        """Verify middlegame tables exist for all piece types."""
        required_pieces = [
            chess.PAWN,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
            chess.KING,
        ]

        for piece_type in required_pieces:
            assert piece_type in PIECE_SQUARE_TABLES_MG, (
                f"Missing MG table for {chess.piece_name(piece_type)}"
            )

    def test_eg_tables_exist_for_all_pieces(self):
        """Verify endgame tables exist for all piece types."""
        required_pieces = [
            chess.PAWN,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
            chess.KING,
        ]

        for piece_type in required_pieces:
            assert piece_type in PIECE_SQUARE_TABLES_EG, (
                f"Missing EG table for {chess.piece_name(piece_type)}"
            )

    def test_tables_have_64_squares(self):
        """Verify all tables have 64 entries."""
        for piece_type, table in PIECE_SQUARE_TABLES_MG.items():
            assert len(table) == 64, (
                f"MG table for {chess.piece_name(piece_type)} has "
                f"{len(table)} entries, expected 64"
            )

        for piece_type, table in PIECE_SQUARE_TABLES_EG.items():
            assert len(table) == 64, (
                f"EG table for {chess.piece_name(piece_type)} has "
                f"{len(table)} entries, expected 64"
            )

    def test_table_values_are_numeric(self):
        """Verify all table values are numeric."""
        for piece_type, table in PIECE_SQUARE_TABLES_MG.items():
            for square, value in enumerate(table):
                assert isinstance(value, (int, float)), (
                    f"MG {chess.piece_name(piece_type)} at {square} is not numeric"
                )

        for piece_type, table in PIECE_SQUARE_TABLES_EG.items():
            for square, value in enumerate(table):
                assert isinstance(value, (int, float)), (
                    f"EG {chess.piece_name(piece_type)} at {square} is not numeric"
                )


class TestPSTValueRanges:
    """Test that PST values are in reasonable ranges."""

    def test_pawn_values_reasonable(self):
        """Verify pawn PST values are in reasonable range."""
        mg_pawn = PIECE_SQUARE_TABLES_MG[chess.PAWN]
        eg_pawn = PIECE_SQUARE_TABLES_EG[chess.PAWN]

        # Pawn PST bonuses should typically be -50 to +50 centipawns
        for value in mg_pawn:
            assert -100 <= value <= 100, (
                f"MG pawn value {value} out of reasonable range"
            )

        for value in eg_pawn:
            assert -100 <= value <= 100, (
                f"EG pawn value {value} out of reasonable range"
            )

    def test_knight_values_reasonable(self):
        """Verify knight PST values are in reasonable range."""
        mg_knight = PIECE_SQUARE_TABLES_MG[chess.KNIGHT]

        # Knight bonuses should be in reasonable range
        for value in mg_knight:
            assert -100 <= value <= 100, (
                f"MG knight value {value} out of reasonable range"
            )

    def test_no_extreme_values(self):
        """Verify no PST values are extremely large."""
        max_value = 200  # No bonus should exceed 2 pawns

        for piece_type, table in PIECE_SQUARE_TABLES_MG.items():
            for value in table:
                assert abs(value) <= max_value, (
                    f"MG {chess.piece_name(piece_type)} has extreme value {value}"
                )

        for piece_type, table in PIECE_SQUARE_TABLES_EG.items():
            for value in table:
                assert abs(value) <= max_value, (
                    f"EG {chess.piece_name(piece_type)} has extreme value {value}"
                )


class TestPSTLogic:
    """Test logical properties of PST values."""

    def test_knights_prefer_center_mg(self):
        """Verify knights have higher values in center (middlegame)."""
        mg_knight = PIECE_SQUARE_TABLES_MG[chess.KNIGHT]

        # Center squares (d4, e4, d5, e5)
        center_squares = [D4, E4, D5, E5]
        # Corner squares
        corner_squares = [A1, H1, A8, H8]

        avg_center = sum(mg_knight[sq] for sq in center_squares) / len(center_squares)
        avg_corner = sum(mg_knight[sq] for sq in corner_squares) / len(corner_squares)

        # Center should be preferred over corners
        assert avg_center > avg_corner, "Knights should prefer center over corners"

    def test_pawns_advance_bonus_mg(self):
        """Verify pawns get bonus for advancing (middlegame)."""
        mg_pawn = PIECE_SQUARE_TABLES_MG[chess.PAWN]

        # Pawns on 7th rank should generally be more valuable than 2nd rank
        # (from White's perspective)
        rank_7 = [A7, B7, C7, D7]
        rank_2 = [A2, B2, C2, D2]

        avg_rank_7 = sum(mg_pawn[sq] for sq in rank_7) / len(rank_7)
        avg_rank_2 = sum(mg_pawn[sq] for sq in rank_2) / len(rank_2)

        # Advanced pawns should generally be more valuable
        assert avg_rank_7 >= avg_rank_2, (
            "Advanced pawns should be at least as valuable as back pawns"
        )

    def test_king_safety_mg(self):
        """Verify king prefers back rank in middlegame."""
        mg_king = PIECE_SQUARE_TABLES_MG[chess.KING]

        # King should prefer back rank in middlegame
        back_rank = [A1, B1, C1, G1, H1]
        center = [D4, E4, D5, E5]

        avg_back = sum(mg_king[sq] for sq in back_rank) / len(back_rank)
        avg_center = sum(mg_king[sq] for sq in center) / len(center)

        # King should prefer back rank in middlegame
        assert avg_back >= avg_center, "King should prefer back rank in middlegame"

    def test_king_active_eg(self):
        """Verify king can be more active in endgame."""
        eg_king = PIECE_SQUARE_TABLES_EG[chess.KING]
        mg_king = PIECE_SQUARE_TABLES_MG[chess.KING]

        # King centralization in endgame
        center = [D4, E4, D5, E5]

        avg_eg_center = sum(eg_king[sq] for sq in center) / len(center)
        avg_mg_center = sum(mg_king[sq] for sq in center) / len(center)

        # King should be relatively more centralized in endgame
        # (or at least not penalized as much)
        assert avg_eg_center >= avg_mg_center, "King should be more active in endgame"


class TestMakeTableFunction:
    """Test the make_table helper function."""

    def test_make_table_creates_64_values(self):
        """Verify make_table creates 64 values from 32."""
        test_array = [0] * 32
        result = make_table(test_array)

        assert len(result) == 64

    def test_make_table_mirrors_values(self):
        """Verify make_table mirrors the first 32 values."""
        test_array = list(range(32))
        result = make_table(test_array)

        assert len(result) == 64
        assert result[0] == 0.0
        assert result[31] == 0.31
        assert result[32] == 0.31  # Mirrored
        assert result[63] == 0.0  # Mirrored

    def test_make_table_converts_to_pawns(self):
        """Verify make_table converts centipawns to pawns."""
        test_array = [100] * 32
        result = make_table(test_array)

        assert all(v == 1.0 for v in result)


class TestPSTConsistency:
    """Test consistency between MG and EG tables."""

    def test_both_tables_have_same_pieces(self):
        """Verify MG and EG tables have the same piece types."""
        mg_pieces = set(PIECE_SQUARE_TABLES_MG.keys())
        eg_pieces = set(PIECE_SQUARE_TABLES_EG.keys())

        assert mg_pieces == eg_pieces, "MG and EG tables should have same piece types"

    def test_table_sizes_match(self):
        """Verify MG and EG tables have same sizes for each piece."""
        for piece_type in PIECE_SQUARE_TABLES_MG:
            mg_size = len(PIECE_SQUARE_TABLES_MG[piece_type])
            eg_size = len(PIECE_SQUARE_TABLES_EG[piece_type])

            assert mg_size == eg_size == 64, (
                f"{chess.piece_name(piece_type)} table size mismatch"
            )


class TestPSTSquareAccess:
    """Test accessing PST values by square."""

    def test_access_all_squares(self):
        """Verify all 64 squares can be accessed."""
        for square in range(64):
            for piece_type in PIECE_SQUARE_TABLES_MG:
                value_mg = PIECE_SQUARE_TABLES_MG[piece_type][square]
                value_eg = PIECE_SQUARE_TABLES_EG[piece_type][square]

                assert isinstance(value_mg, (int, float))
                assert isinstance(value_eg, (int, float))

    def test_corner_squares(self):
        """Verify corner squares can be accessed."""
        corners = [A1, H1, A8, H8]

        for corner in corners:
            for piece_type in PIECE_SQUARE_TABLES_MG:
                value = PIECE_SQUARE_TABLES_MG[piece_type][corner]
                assert isinstance(value, (int, float))

    def test_center_squares(self):
        """Verify center squares can be accessed."""
        center = [D4, E4, D5, E5]

        for square in center:
            for piece_type in PIECE_SQUARE_TABLES_MG:
                value = PIECE_SQUARE_TABLES_MG[piece_type][square]
                assert isinstance(value, (int, float))
