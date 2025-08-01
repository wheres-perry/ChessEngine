"""
Comprehensive edge case tests for chess move generation.
These tests target specific bugs and edge cases that commonly break chess engines.
"""

# ruff: noqa
import pytest

from src.engine._core import chess_engine_core as core  # type: ignore


def verify_legal_moves(
    fen: str,
    expected_count: int,
    expected_moves: list[tuple[int, int, int]],
    message: str = "Legal moves do not match expected",
) -> None:
    """Helper to verify legal moves for a given FEN."""
    board = core.Board.from_fen(fen)
    moves = board.generate_legal_moves()

    # Print for debugging
    print(core.moves_to_string(moves, board))

    # Assert count
    assert (
        len(moves) == expected_count
    ), f"Expected {expected_count} legal moves, but found {len(moves)}"

    # Convert generated moves to tuples
    generated_tuples = [(m.from_square, m.to, m.promotion) for m in moves]

    # Sort for order-independent comparison
    expected_moves_sorted = sorted(expected_moves)
    generated_tuples_sorted = sorted(generated_tuples)

    assert generated_tuples_sorted == expected_moves_sorted, message


class TestMoveGenerationEdgeCases:
    """Bulletproof tests for move generation edge cases."""

    def test_prevented_castle(self):
        """Test position where castling is prevented,only 4 specific moves are legal."""
        fen = "4k3/8/8/8/1bb5/4P3/3P3r/r2NK2R w K - 0 1"
        expected_moves: list[tuple[int, int, int]] = [
            (20, 28, 0),  # Pawn e3 to e4
            (7, 5, 0),  # Rook h1 to f1
            (7, 6, 0),  # Rook h1 to g1
            (7, 15, 0),  # Rook h1 to h2
        ]
        verify_legal_moves(
            fen,
            expected_count=4,
            expected_moves=expected_moves,
            message="Generated moves do not match expected (prevented castling)",
        )

    def test_forced_en_passant(self):
        """Test position where en passant is the only legal move (forced)."""
        fen = "K7/8/8/8/1pP5/1P4Q1/8/7k b - c3 0 1"
        expected_moves: list[tuple[int, int, int]] = [
            (25, 18, 0),  # Black pawn b4 captures en passant on c3
        ]
        verify_legal_moves(
            fen,
            expected_count=1,
            expected_moves=expected_moves,
            message="Only en passant capture should be legal in this forced position",
        )

    def test_forced_stalemate_prevent_en_passant(self):
        """Test position where it's stalemate (0 legal moves) and
        en passant is prevented due to pinned pawn."""
        fen = "6k1/7p/5Q1B/8/2pP4/2P5/B7/6K1 b - - 0 1"
        expected_moves: list[tuple[int, int, int]] = []
        verify_legal_moves(
            fen,
            expected_count=0,
            expected_moves=expected_moves,
            message=(
                "Should be stalemate with 0 legal moves "
                "(en passant prevented due to pin)"
            ),
        )

    def test_pinned_en_passant_force_capture(self):
        """Test position where en passant is pinned and only legal move is king to b4"""
        fen = "8/7p/5R1B/2k5/BPp5/3Q4/8/2R3K1 b - - 0 1"
        expected_moves: list[tuple[int, int, int]] = [
            (34, 25, 0),  # King c5 to b4
        ]
        verify_legal_moves(
            fen,
            expected_count=1,
            expected_moves=expected_moves,
            message="Only king move to b4 should be legal (en passant pinned)",
        )

    def test_pin_prevent_capture_forced_king(self):
        """Test position where pin prevents capture and
        forces specific king move (only 1 legal move)."""
        fen = "4k3/6P1/7q/8/6K1/4n3/4R2b/3b4 w - - 1 2"
        expected_moves: list[tuple[int, int, int]] = [
            (30, 21, 0),  # King g4 to f3
        ]
        verify_legal_moves(
            fen,
            expected_count=1,
            expected_moves=expected_moves,
            message=(
                "Only king move g4 to f3 should be legal (pin prevents other captures)"
            ),
        )

    def test_pin_force_king_capture(self):
        """Test position where the only move is king to e2 forced capture due to pin"""
        fen = "4k3/8/8/8/8/8/4q3/r2QK3 w - - 0 1"
        expected_moves: list[tuple[int, int, int]] = [
            (4, 12, 0),  # King e1 to e2
        ]
        verify_legal_moves(
            fen,
            expected_count=1,
            expected_moves=expected_moves,
            message=(
                "Only king move e1 to e2 should be legal (forced capture due to pin)"
            ),
        )

    def test_pin_prevent_capture_checkmate(self):
        """Test position where pin prevents capture, resulting in checkmate (0 legal moves)."""
        fen = "4k3/6P1/7q/8/6K1/4nR2/7b/3b4 w - - 1 2"
        expected_moves: list[tuple[int, int, int]] = []
        verify_legal_moves(
            fen,
            expected_count=0,
            expected_moves=expected_moves,
            message="Should be checkmate with 0 legal moves (pin prevents capture)",
        )

    def test_forced_king_prevent_en_passant(self):
        """Test position where king is forced to move (only 1 legal move)."""
        fen = "6k1/8/5Q2/8/2pP4/2P5/B7/6K1 b - d3 0 1"
        expected_moves: list[tuple[int, int, int]] = [
            (62, 55, 0),  # King g8 to h7
        ]
        verify_legal_moves(
            fen,
            expected_count=1,
            expected_moves=expected_moves,
            message="Only king move to h7 should be legal (en passant prevented)",
        )

    def test_lost_castle_rights(self):
        """Test position where castling rights are lost 4 specific moves are legal."""
        fen = "4k3/8/8/8/1b2p3/4P3/3P3r/r2NK2R w - - 0 1"
        expected_moves: list[tuple[int, int, int]] = [
            (7, 15, 0),  # Rook h1 to h2
            (7, 6, 0),  # Rook h1 to g1
            (7, 5, 0),  # Rook h1 to f1
            (4, 5, 0),  # King e1 to f1
        ]
        verify_legal_moves(
            fen,
            expected_count=4,
            expected_moves=expected_moves,
            message="Generated moves do not match expected after lost castle rights",
        )

        # Additional assertion for lost rights (if needed)
        board = core.Board.from_fen(fen)
        assert board.get_castling_rights() == 0, "Castling rights should be lost"

    def test_expired_en_passant_stalemate(self):
        """Test position where en passant has expired resulting in stalemate (0 legal moves)."""
        fen = "4k3/8/5Q2/5B2/1Pp5/2P5/4K3/8 b - - 1 2"
        expected_moves: list[tuple[int, int, int]] = []
        verify_legal_moves(
            fen,
            expected_count=0,
            expected_moves=expected_moves,
            message="Should be stalemate with 0 legal moves (en passant expired)",
        )

    def test_expired_en_passant(self):
        """Test position where en passant has expired and only pawn c4 to c3 is legal (1 move)."""
        fen = "4k3/8/5Q2/5B2/1Pp5/8/4K3/8 b - - 1 2"
        expected_moves: list[tuple[int, int, int]] = [
            (26, 18, 0),  # Pawn c4 to c3
        ]
        verify_legal_moves(
            fen,
            expected_count=1,
            expected_moves=expected_moves,
            message="Only pawn move c4 to c3 should be legal (en passant expired)",
        )

    def test_no_queen_castle_thru_attacker(self):
        """Test position where queenside castling is prevented due to attacker and only 3 rook moves are legal."""
        fen = "r3k3/p7/P4Q2/5B2/8/8/8/5K2 b - - 0 1"
        expected_moves: list[tuple[int, int, int]] = [
            (56, 57, 0),  # Rook a8 to b8
            (56, 58, 0),  # Rook a8 to c8
            (56, 59, 0),  # Rook a8 to d8
        ]
        verify_legal_moves(
            fen,
            expected_count=3,
            expected_moves=expected_moves,
            message=(
                "Only 3 rook moves from a8 should be legal "
                "(queenside castling prevented due to attacker)"
            ),
        )

    def test_no_castle_checkmate(self):
        """Test position where it's checkmate and castling is not possible (0 legal moves)."""
        fen = "6k1/8/8/8/8/2bq1p2/5P1P/4K2R w K - 0 1"
        expected_moves: list[tuple[int, int, int]] = []
        verify_legal_moves(
            fen,
            expected_count=0,
            expected_moves=expected_moves,
            message="Should be checkmate with 0 legal moves (castling not possible)",
        )

    def test_no_queen_castle_thru_piece_forced_king(self):
        """Test position where queenside castling is blocked by piece and king is forced to d8 (only 1 legal move)."""
        fen = "rb2k3/p1p1p1Q1/P1P1P3/8/8/8/8/5K2 b - - 0 1"
        expected_moves: list[tuple[int, int, int]] = [
            (60, 59, 0),  # King e8 to d8
        ]
        print("Expected moves:")
        # Convert tuples to Move objects
        board = core.Board.from_fen(fen)
        move_objects = [
            core.Move(from_sq, to_sq, promo) for from_sq, to_sq, promo in expected_moves
        ]
        print(core.moves_to_string(move_objects, board))

        verify_legal_moves(
            fen,
            expected_count=1,
            expected_moves=expected_moves,
            message=(
                "Only king move e8 to d8 should be legal "
                "(queenside castling blocked by piece)"
            ),
        )

    def test_no_queen_castle_thru_piece_or_attack_stalemate(self):
        """Test position where queenside castling is prevented through piece or attack resulting in stalemate (0 legal moves)."""
        fen = "rb2k3/p1p1p1Q1/P1P1P3/8/8/8/8/3R1K2 b - - 0 1"
        expected_moves: list[tuple[int, int, int]] = []
        print("Expected moves:")
        # Convert tuples to Move objects
        board = core.Board.from_fen(fen)
        move_objects = [
            core.Move(from_sq, to_sq, promo) for from_sq, to_sq, promo in expected_moves
        ]
        print(core.moves_to_string(move_objects, board))

        verify_legal_moves(
            fen,
            expected_count=0,
            expected_moves=expected_moves,
            message=(
                "Should be stalemate with 0 legal moves "
                "(cannot queenside castle through piece or attack)"
            ),
        )

    def test_double_pinned_en_passant_stalemate(self):
        """Test position with double pinned en passant resulting in stalemate (0 legal moves)."""
        fen = "8/2Q1Q3/8/3k4/2pPp3/8/B5B1/3R1K2 b - - 0 1"
        expected_moves: list[tuple[int, int, int]] = []
        verify_legal_moves(
            fen,
            expected_count=0,
            expected_moves=expected_moves,
            message=(
                "Should be stalemate with 0 legal moves (double pinned en passant)"
            ),
        )

    def test_double_en_passant(self):
        """Test position with double en passant opportunity (only 2 legal moves)."""
        fen = "8/2Q1Q3/8/3k4/2pPp3/2P1P3/8/3R1K2 b - d3 0 1"
        expected_moves: list[tuple[int, int, int]] = [
            (28, 19, 0),  # Pawn e4 to d3 (en passant)
            (26, 19, 0),  # Pawn c4 to d3 (en passant)
        ]
        verify_legal_moves(
            fen,
            expected_count=2,
            expected_moves=expected_moves,
            message=(
                "Only two en passant captures should be legal in double en "
                "passant position"
            ),
        )

    def test_forced_king_capture_pin(self):
        """Test position where pin forces king capture and only legal move is d2 to c3."""
        fen = "6k1/8/8/8/2b5/2b3p1/2pB4/4K3 w - - 0 1"
        expected_moves: list[tuple[int, int, int]] = [
            (11, 18, 0),  # d2 to c3
        ]
        verify_legal_moves(
            fen,
            expected_count=1,
            expected_moves=expected_moves,
            message=(
                "Only move d2 to c3 should be legal due to forced king capture with pin"
            ),
        )

    def test_basic_promotion(self):
        """Test position where legal moves are pawn promotion to all piece types plus one king move."""
        fen = "8/P7/8/8/8/6r1/5k2/7K w - - 0 1"
        expected_moves = [
            (48, 56, 4),  # Pawn a7 to a8 promote to Queen
            (48, 56, 3),  # Pawn a7 to a8 promote to Rook
            (48, 56, 2),  # Pawn a7 to a8 promote to Bishop
            (48, 56, 1),  # Pawn a7 to a8 promote to Knight
            (7, 15, 0),  # King h1 to h2
        ]
        verify_legal_moves(
            fen,
            expected_count=5,
            expected_moves=expected_moves,
            message="Should have 4 pawn promotion moves plus 1 king move",
        )

    def test_pinned_promotion_force_capture(self):
        """Test position where pinned pawn must capture and promote (only 4 legal moves)."""
        fen = "q1q4k/1P6/8/8/8/6q1/8/7K w - - 0 1"
        expected_moves = [
            (49, 56, 4),  # Pawn b7 captures a8 promote to Queen
            (49, 56, 3),  # Pawn b7 captures a8 promote to Rook
            (49, 56, 2),  # Pawn b7 captures a8 promote to Bishop
            (49, 56, 1),  # Pawn b7 captures a8 promote to Knight
        ]
        verify_legal_moves(
            fen,
            expected_count=4,
            expected_moves=expected_moves,
            message="Only pawn capture promotions b7xa8 should be legal due to pin",
        )

    def test_triple_possible_promotion(self):
        """Test position where pawn can promote to 3 different squares with all piece types (12 legal moves)."""
        fen = "1q1q3k/2P5/8/8/8/6q1/8/7K w - - 0 1"
        expected_moves = [
            # c7 to b8 promotions
            (50, 57, 4),  # Pawn c7 to b8 promote to Queen
            (50, 57, 3),  # Pawn c7 to b8 promote to Rook
            (50, 57, 2),  # Pawn c7 to b8 promote to Bishop
            (50, 57, 1),  # Pawn c7 to b8 promote to Knight
            # c7 to c8 promotions
            (50, 58, 4),  # Pawn c7 to c8 promote to Queen
            (50, 58, 3),  # Pawn c7 to c8 promote to Rook
            (50, 58, 2),  # Pawn c7 to c8 promote to Bishop
            (50, 58, 1),  # Pawn c7 to c8 promote to Knight
            # c7 to d8 promotions
            (50, 59, 4),  # Pawn c7 to d8 promote to Queen
            (50, 59, 3),  # Pawn c7 to d8 promote to Rook
            (50, 59, 2),  # Pawn c7 to d8 promote to Bishop
            (50, 59, 1),  # Pawn c7 to d8 promote to Knight
        ]
        verify_legal_moves(
            fen,
            expected_count=12,
            expected_moves=expected_moves,
            message="Should have 12 legal moves: 3 promotion squares × 4 piece types each",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
