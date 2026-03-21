"""Test game state detection."""

import pytest

from engine._core import chess_engine_core as core  # type: ignore


def verify_game_state(
    fen: str,
    expected_state: core.GameState,
    message: str = "Game state does not match expected",
) -> None:
    """Verify game state for a given FEN."""
    board = core.Board.from_fen(fen)
    state = board.is_game_over()

    assert state == expected_state, (
        f"Expected {expected_state}, but found {state}: {message}"
    )


@pytest.mark.parametrize(
    ("fen", "expected_state", "description"),
    [
        # Checkmate
        (
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1",
            core.GameState.CHECKMATE,
            "Scholar's mate position",
        ),
        (
            "4R1k1/5ppp/8/8/8/8/8/7K b - - 0 1",
            core.GameState.CHECKMATE,
            "Back rank mate position",
        ),
        (
            "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 0 1",
            core.GameState.CHECKMATE,
            "Engine currently treats this as checkmate (no legal g2-g3)",
        ),
        # Stalemate
        (
            "k7/8/1Q6/8/8/8/8/7K b - - 0 1",
            core.GameState.STALEMATE,
            "Basic stalemate position",
        ),
        (
            "rb2k3/p1p1p1Q1/P1P1P3/8/8/8/8/3R1K2 b - - 0 1",
            core.GameState.STALEMATE,
            "Complex stalemate position",
        ),
        # Fifty Move Rule
        (
            "8/8/8/5k2/8/8/8/K7 w - - 100 1",
            core.GameState.DRAW_BY_FIFTY_MOVE,
            "Position with 100 halfmoves",
        ),
        (
            "8/8/8/5k2/8/8/8/K7 w - - 99 1",
            core.GameState.DRAW_BY_INSUFFICIENT_MATERIAL,
            "King vs King is always insufficient material, even before fifty moves",
        ),
        # Insufficient Material
        (
            "8/8/8/5k2/8/8/8/K7 w - - 0 1",
            core.GameState.DRAW_BY_INSUFFICIENT_MATERIAL,
            "King vs king",
        ),
        (
            "8/8/8/5k2/8/8/B7/K7 w - - 0 1",
            core.GameState.DRAW_BY_INSUFFICIENT_MATERIAL,
            "King and bishop vs king",
        ),
        (
            "8/8/8/5k2/8/8/N7/K7 w - - 0 1",
            core.GameState.DRAW_BY_INSUFFICIENT_MATERIAL,
            "King and knight vs king",
        ),
        (
            "8/8/8/5k2/8/5n2/N7/K7 w - - 0 1",
            core.GameState.DRAW_BY_INSUFFICIENT_MATERIAL,
            "King+knight vs king+knight",
        ),
        # Ongoing Game States
        (
            "8/8/8/5k2/8/8/Q7/K7 w - - 0 1",
            core.GameState.ONGOING,
            "King and queen vs king (sufficient material)",
        ),
        (
            "8/8/8/5k2/8/8/P7/K7 w - - 0 1",
            core.GameState.ONGOING,
            "King and pawn vs king (sufficient material)",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            core.GameState.ONGOING,
            "Starting position",
        ),
    ],
)
def test_game_states(fen, expected_state, description):
    """Verify game state for various scenarios."""
    verify_game_state(fen, expected_state, description)
