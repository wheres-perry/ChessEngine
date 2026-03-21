"""Test chess move generation edge cases and bug scenarios."""

import pytest

from engine._core import chess_engine_core as core  # type: ignore


def _move_to_tuple(move: core.Move) -> tuple[int, int, int]:
    """Normalize a move object into a tuple for comparison in assertions."""
    to_square: int | None = getattr(move, "to", None)
    if to_square is None:
        to_square = getattr(move, "to_square", None)

    if to_square is None:
        raise AttributeError(
            "Move objects must expose either 'to' or 'to_square' attributes."
        )

    return move.from_square, to_square, move.promotion


def verify_legal_moves(
    fen: str,
    expected_count: int,
    expected_moves: list[tuple[int, int, int]],
    message: str = "Legal moves do not match expected",
) -> None:
    """Verify legal moves for a given FEN."""
    board = core.Board.from_fen(fen)
    moves = board.generate_legal_moves()

    assert len(moves) == expected_count, (
        f"Expected {expected_count} legal moves, but found {len(moves)}\n{message}"
    )

    generated_tuples = [_move_to_tuple(move) for move in moves]

    expected_moves_sorted = sorted(expected_moves)
    generated_tuples_sorted = sorted(generated_tuples)

    assert generated_tuples_sorted == expected_moves_sorted, message


@pytest.mark.parametrize(
    ("fen", "expected_count", "expected_moves", "description"),
    [
        # Castling Edge Cases
        (
            "4k3/8/8/8/1bb5/4P3/3P3r/r2NK2R w K - 0 1",
            4,
            [(20, 28, 0), (7, 5, 0), (7, 6, 0), (7, 15, 0)],
            "Prevented castle",
        ),
        (
            "4k3/8/8/8/1b2p3/4P3/3P3r/r2NK2R w - - 0 1",
            4,
            [(7, 15, 0), (7, 6, 0), (7, 5, 0), (4, 5, 0)],
            "Lost castle rights",
        ),
        (
            "r3k3/p7/P4Q2/5B2/8/8/8/5K2 b - - 0 1",
            3,
            [(56, 57, 0), (56, 58, 0), (56, 59, 0)],
            "No queen castle thru attacker",
        ),
        (
            "rb2k3/p1p1p1Q1/P1P1P3/8/8/8/8/5K2 b - - 0 1",
            1,
            [(60, 59, 0)],
            "No queen castle thru piece, forced king",
        ),
        (
            "rb2k3/p1p1p1Q1/P1P1P3/8/8/8/8/3R1K2 b - - 0 1",
            0,
            [],
            "No queen castle thru piece or attack, stalemate",
        ),
        (
            "6k1/8/8/8/8/2bq1p2/5P1P/4K2R w K - 0 1",
            0,
            [],
            "No castle, checkmate",
        ),
        # En Passant Edge Cases
        (
            "K7/8/8/8/1pP5/1P4Q1/8/7k b - c3 0 1",
            1,
            [(25, 18, 0)],
            "Forced en passant",
        ),
        (
            "6k1/7p/5Q1B/8/2pP4/2P5/B7/6K1 b - - 0 1",
            0,
            [],
            "Forced stalemate prevent en passant",
        ),
        (
            "8/7p/5R1B/2k5/BPp5/3Q4/8/2R3K1 b - - 0 1",
            1,
            [(34, 25, 0)],
            "Pinned en passant force capture",
        ),
        (
            "6k1/8/5Q2/8/2pP4/2P5/B7/6K1 b - d3 0 1",
            1,
            [(62, 55, 0)],
            "Forced king prevent en passant",
        ),
        (
            "4k3/8/5Q2/5B2/1Pp5/2P5/4K3/8 b - - 1 2",
            0,
            [],
            "Expired en passant stalemate",
        ),
        (
            "4k3/8/5Q2/5B2/1Pp5/8/4K3/8 b - - 1 2",
            1,
            [(26, 18, 0)],
            "Expired en passant",
        ),
        (
            "8/2Q1Q3/8/3k4/2pPp3/8/B5B1/3R1K2 b - - 0 1",
            0,
            [],
            "Double pinned en passant stalemate",
        ),
        (
            "8/2Q1Q3/8/3k4/2pPp3/2P1P3/8/3R1K2 b - d3 0 1",
            2,
            [(28, 19, 0), (26, 19, 0)],
            "Double en passant",
        ),
        # Pin and Check Scenarios
        (
            "4k3/6P1/7q/8/6K1/4n3/4R2b/3b4 w - - 1 2",
            1,
            [(30, 21, 0)],
            "Pin prevent capture, forced king",
        ),
        (
            "4k3/8/8/8/8/8/4q3/r2QK3 w - - 0 1",
            1,
            [(4, 12, 0)],
            "Pin force king capture",
        ),
        (
            "4k3/6P1/7q/8/6K1/4nR2/7b/3b4 w - - 1 2",
            0,
            [],
            "Pin prevent capture checkmate",
        ),
        (
            "6k1/8/8/8/2b5/2b3p1/2pB4/4K3 w - - 0 1",
            1,
            [(11, 18, 0)],
            "Forced king capture pin",
        ),
        # Pawn Promotion Scenarios
        (
            "8/P7/8/8/8/6r1/5k2/7K w - - 0 1",
            5,
            [(48, 56, 4), (48, 56, 3), (48, 56, 2), (48, 56, 1), (7, 15, 0)],
            "Basic promotion",
        ),
        (
            "q1q4k/1P6/8/8/8/6q1/8/7K w - - 0 1",
            4,
            [(49, 56, 4), (49, 56, 3), (49, 56, 2), (49, 56, 1)],
            "Pinned promotion force capture",
        ),
        (
            "1q1q3k/2P5/8/8/8/6q1/8/7K w - - 0 1",
            12,
            [
                (50, 57, 4),
                (50, 57, 3),
                (50, 57, 2),
                (50, 57, 1),
                (50, 58, 4),
                (50, 58, 3),
                (50, 58, 2),
                (50, 58, 1),
                (50, 59, 4),
                (50, 59, 3),
                (50, 59, 2),
                (50, 59, 1),
            ],
            "Triple possible promotion",
        ),
        # Piece Movement Patterns
        (
            "k6r/8/3pPP2/3PQP2/3PP3/6P1/4q3/6K1 w - - 0 1",
            5,
            [(44, 52, 0), (45, 53, 0), (36, 29, 0), (36, 43, 0), (22, 30, 0)],
            "Queen diagonal move and capture",
        ),
        (
            "k6r/8/3pPP2/3PBP2/3PP3/6P1/4q3/6K1 w - - 0 1",
            5,
            [(44, 52, 0), (45, 53, 0), (36, 29, 0), (36, 43, 0), (22, 30, 0)],
            "Bishop diagonal move and capture",
        ),
        (
            "5q2/1k1K1P2/3P1P2/3PQP2/3P1P2/8/4n3/8 w - - 0 1",
            7,
            [
                (36, 60, 0),
                (36, 52, 0),
                (36, 44, 0),
                (36, 28, 0),
                (36, 20, 0),
                (36, 12, 0),
                (51, 44, 0),
            ],
            "Queen orthogonal move and capture",
        ),
        (
            "5q2/1k1K1P2/3P1P2/3PRP2/3P1P2/8/4n3/8 w - - 0 1",
            7,
            [
                (36, 60, 0),
                (36, 52, 0),
                (36, 44, 0),
                (36, 28, 0),
                (36, 20, 0),
                (36, 12, 0),
                (51, 44, 0),
            ],
            "Rook orthogonal move and capture",
        ),
    ],
)
def test_move_generation_edge_cases(fen, expected_count, expected_moves, description):
    """Verify various move generation edge cases using parameterization."""
    verify_legal_moves(fen, expected_count, expected_moves, description)


def test_lost_castle_rights_state():
    """Verify state is correctly updated after lost castle rights."""
    fen = "4k3/8/8/8/1b2p3/4P3/3P3r/r2NK2R w - - 0 1"
    board = core.Board.from_fen(fen)
    assert board.get_castling_rights() == 0, "Castling rights should be lost"
