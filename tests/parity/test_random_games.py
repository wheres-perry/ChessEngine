"""Extensive parity tests comparing C++ engine against python-chess across random games.

Simulates 100 random games, making random moves and verifying that:
1. Legal move generation matches at every position
2. Board state (FEN) matches after each move
3. Game termination conditions match
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import chess as pychess
import pytest

from engine._core import chess_engine_core as core

# Test configuration
NUM_GAMES = 100
MAX_MOVES_PER_GAME = 20
RANDOM_SEED = 42


def get_cpp_legal_moves_uci(board: core.Board) -> set[str]:
    """Return legal moves from C++ board as UCI strings."""
    return {core.move_to_uci(m) for m in board.generate_legal_moves()}


def get_pychess_legal_moves_uci(board: pychess.Board) -> set[str]:
    """Return legal moves from python-chess board as UCI strings."""
    return {m.uci() for m in board.legal_moves}


def compare_fen_ignore_ep(cpp_fen: str, py_fen: str) -> bool:
    """Compare FEN strings ignoring the en-passant square field.

    Known difference: C++ engine sets ep square on any double pawn push,
    python-chess only sets it when an en passant capture is actually possible.

    Args:
        cpp_fen: FEN string from C++ engine.
        py_fen: FEN string from python-chess.

    Returns:
        True if FENs match (excluding ep square), False otherwise.

    """
    cpp_parts = cpp_fen.split()
    py_parts = py_fen.split()

    # cpp_parts[3] and py_parts[3] are the ep squares (excluded from comparison)
    return (
        cpp_parts[0] == py_parts[0]  # Position
        and cpp_parts[1] == py_parts[1]  # Side to move
        and cpp_parts[2] == py_parts[2]  # Castling rights
        and cpp_parts[4] == py_parts[4]  # Halfmove clock
        and cpp_parts[5] == py_parts[5]  # Fullmove number
    )


class TestRandomGameParity:
    """Test legal move parity across many random games."""

    @pytest.fixture(autouse=True)
    def setup_rng(self) -> None:
        """Set up deterministic random number generator."""
        random.seed(RANDOM_SEED)

    def test_100_random_games_legal_moves_parity(self) -> None:
        """Play 100 random games, verifying legal move parity at each position.

        For each game:
        1. Start from initial position
        2. Pick a random legal move from C++ board
        3. Apply to both C++ and python-chess boards
        4. Verify legal moves match
        5. Repeat for up to 20 moves or until game over
        """
        games_completed = 0
        total_positions_checked = 0
        games_ended_early = 0

        for game_num in range(NUM_GAMES):
            cpp_board = core.Board()
            py_board = pychess.Board()

            game_over = False

            for move_num in range(MAX_MOVES_PER_GAME):
                cpp_moves = get_cpp_legal_moves_uci(cpp_board)
                py_moves = get_pychess_legal_moves_uci(py_board)

                if cpp_moves != py_moves:
                    only_in_cpp = cpp_moves - py_moves
                    only_in_py = py_moves - cpp_moves

                    pytest.fail(
                        f"Legal move mismatch in game {game_num + 1}, "
                        f"move {move_num + 1}!\n"
                        f"FEN: {cpp_board.fen()}\n"
                        f"Only in C++: {sorted(only_in_cpp)}\n"
                        f"Only in python-chess: {sorted(only_in_py)}\n"
                        f"C++ has {len(cpp_moves)} moves, python-chess has "
                        f"{len(py_moves)} moves"
                    )

                total_positions_checked += 1

                if not cpp_moves:
                    game_over = True
                    games_ended_early += 1
                    break

                chosen_uci = random.choice(sorted(cpp_moves))

                cpp_board.push(core.Move.from_uci(chosen_uci))
                py_board.push(pychess.Move.from_uci(chosen_uci))

                cpp_fen_parts = cpp_board.fen().split()
                py_fen_parts = py_board.fen().split()

                if (
                    cpp_fen_parts[0] != py_fen_parts[0]
                    or cpp_fen_parts[1] != py_fen_parts[1]
                    or cpp_fen_parts[2] != py_fen_parts[2]
                ):
                    pytest.fail(
                        f"Board state mismatch after move {chosen_uci} in "
                        f"game {game_num + 1}!\n"
                        f"C++ FEN:  {cpp_board.fen()}\n"
                        f"Py FEN:   {py_board.fen()}"
                    )

            games_completed += 1

        print(f"\n✓ Completed {games_completed} games")
        print(f"✓ Checked {total_positions_checked} positions")
        print(f"✓ {games_ended_early} games ended before {MAX_MOVES_PER_GAME} moves")

        assert games_completed == NUM_GAMES

    def test_random_games_with_full_fen_comparison(self) -> None:
        """Verify FEN parity including halfmove clock and fullmove number.

        Uses fewer games since this is stricter.

        Note: En passant square is excluded from comparison due to known difference -
        C++ engine sets ep square on any double pawn push, python-chess only when
        an en passant capture is actually possible.
        """
        num_games = 20

        for _game_num in range(num_games):
            cpp_board = core.Board()
            py_board = pychess.Board()

            for move_num in range(MAX_MOVES_PER_GAME):
                cpp_moves = get_cpp_legal_moves_uci(cpp_board)
                py_moves = get_pychess_legal_moves_uci(py_board)

                assert cpp_moves == py_moves, (
                    f"Move mismatch at game {_game_num + 1}, move {move_num + 1}"
                )

                if not cpp_moves:
                    break

                chosen_uci = random.choice(sorted(cpp_moves))

                cpp_board.push(core.Move.from_uci(chosen_uci))
                py_board.push(pychess.Move.from_uci(chosen_uci))

                assert compare_fen_ignore_ep(cpp_board.fen(), py_board.fen()), (
                    f"FEN mismatch after {chosen_uci}:\n"
                    f"C++: {cpp_board.fen()}\n"
                    f"Py:  {py_board.fen()}"
                )

    def test_random_games_check_detection(self) -> None:
        """Verify check detection matches across random games."""
        num_games = 50
        checks_found = 0

        for _game_num in range(num_games):
            cpp_board = core.Board()
            py_board = pychess.Board()

            for _ in range(MAX_MOVES_PER_GAME):
                cpp_moves = get_cpp_legal_moves_uci(cpp_board)

                if not cpp_moves:
                    break

                chosen_uci = random.choice(sorted(cpp_moves))

                cpp_board.push(core.Move.from_uci(chosen_uci))
                py_board.push(pychess.Move.from_uci(chosen_uci))

                cpp_in_check = cpp_board.is_check()
                py_in_check = py_board.is_check()

                if cpp_in_check:
                    checks_found += 1

                assert cpp_in_check == py_in_check, (
                    f"Check detection mismatch!\n"
                    f"FEN: {cpp_board.fen()}\n"
                    f"C++ says in_check={cpp_in_check}, python-chess says {py_in_check}"
                )

        print(f"\n✓ Found {checks_found} check positions across {num_games} games")

    def test_random_games_capture_detection(self) -> None:
        """Verify capture detection matches for all legal moves."""
        num_games = 30
        captures_checked = 0

        for _game_num in range(num_games):
            cpp_board = core.Board()
            py_board = pychess.Board()

            for _ in range(MAX_MOVES_PER_GAME):
                for move in cpp_board.generate_legal_moves():
                    uci = core.move_to_uci(move)
                    py_move = pychess.Move.from_uci(uci)

                    cpp_is_capture = cpp_board.is_capture(move)
                    py_is_capture = py_board.is_capture(py_move)

                    if cpp_is_capture != py_is_capture:
                        pytest.fail(
                            f"Capture detection mismatch for {uci}!\n"
                            f"FEN: {cpp_board.fen()}\n"
                            f"C++: {cpp_is_capture}, python-chess: {py_is_capture}"
                        )

                    captures_checked += 1

                cpp_moves = list(cpp_board.generate_legal_moves())
                if not cpp_moves:
                    break

                chosen = random.choice(cpp_moves)
                chosen_uci = core.move_to_uci(chosen)

                cpp_board.push(chosen)
                py_board.push(pychess.Move.from_uci(chosen_uci))

        print(f"\n✓ Checked capture status for {captures_checked} moves")

    def test_random_games_castling_detection(self) -> None:
        """Verify castling move detection matches."""
        num_games = 50
        castling_moves_found = 0

        for _game_num in range(num_games):
            cpp_board = core.Board()
            py_board = pychess.Board()

            for _ in range(MAX_MOVES_PER_GAME):
                for move in cpp_board.generate_legal_moves():
                    uci = core.move_to_uci(move)
                    py_move = pychess.Move.from_uci(uci)

                    cpp_is_castling = cpp_board.is_castling(move)
                    py_is_castling = py_board.is_castling(py_move)

                    if cpp_is_castling:
                        castling_moves_found += 1

                    if cpp_is_castling != py_is_castling:
                        pytest.fail(
                            f"Castling detection mismatch for {uci}!\n"
                            f"FEN: {cpp_board.fen()}\n"
                            f"C++: {cpp_is_castling}, python-chess: {py_is_castling}"
                        )

                cpp_moves = list(cpp_board.generate_legal_moves())
                if not cpp_moves:
                    break

                chosen = random.choice(cpp_moves)
                chosen_uci = core.move_to_uci(chosen)

                cpp_board.push(chosen)
                py_board.push(pychess.Move.from_uci(chosen_uci))

        print(
            f"\n✓ Found {castling_moves_found} castling moves across {num_games} games"
        )


class TestSpecificPositionsParity:
    """Test parity on specific tricky positions."""

    @pytest.mark.parametrize(
        "fen",
        [
            # Complex middlegame positions
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            # KiwiPete
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            # Positions with en passant available
            "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3",
            # Positions with limited castling rights
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kq - 0 1",
            # Promotion positions
            "8/P7/8/8/8/8/8/4K2k w - - 0 1",
            "8/8/8/8/8/8/p7/4K2k b - - 0 1",
            # Endgame positions
            "8/8/8/2k5/8/8/3K4/8 w - - 0 1",
            "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
        ],
    )
    def test_position_legal_moves_parity(self, fen: str) -> None:
        """Test legal move parity on specific positions."""
        cpp_board = core.Board.from_fen(fen)
        py_board = pychess.Board(fen)

        cpp_moves = get_cpp_legal_moves_uci(cpp_board)
        py_moves = get_pychess_legal_moves_uci(py_board)

        assert cpp_moves == py_moves, (
            f"Legal move mismatch for FEN: {fen}\n"
            f"Only in C++: {sorted(cpp_moves - py_moves)}\n"
            f"Only in python-chess: {sorted(py_moves - cpp_moves)}"
        )

    def test_en_passant_positions(self) -> None:
        """Test en passant capture scenarios."""
        cpp_board = core.Board()
        py_board = pychess.Board()

        moves = ["e4", "a6", "e5", "d5"]  # d5 creates en passant on d6

        for san in moves:
            cpp_board.push_san(san)
            py_board.push_san(san)

        cpp_moves = get_cpp_legal_moves_uci(cpp_board)
        py_moves = get_pychess_legal_moves_uci(py_board)

        assert "e5d6" in cpp_moves, "C++ should have en passant move"
        assert "e5d6" in py_moves, "python-chess should have en passant move"
        assert cpp_moves == py_moves

    def test_promotion_moves(self) -> None:
        """Test that promotion moves are generated correctly."""
        fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"

        cpp_board = core.Board.from_fen(fen)
        py_board = pychess.Board(fen)

        cpp_moves = get_cpp_legal_moves_uci(cpp_board)
        py_moves = get_pychess_legal_moves_uci(py_board)

        promotion_moves = {m for m in cpp_moves if m.startswith("a7a8")}
        assert len(promotion_moves) == 4, (
            f"Expected 4 promotion moves, got {promotion_moves}"
        )

        assert cpp_moves == py_moves


class TestGameTerminationParity:
    """Test game termination detection parity."""

    def test_checkmate_detection(self) -> None:
        """Test checkmate is detected correctly."""
        cpp_board = core.Board()
        py_board = pychess.Board()

        moves = ["f3", "e5", "g4", "Qh4"]
        for san in moves:
            cpp_board.push_san(san)
            py_board.push_san(san)

        cpp_moves = get_cpp_legal_moves_uci(cpp_board)
        py_moves = get_pychess_legal_moves_uci(py_board)

        assert len(cpp_moves) == 0, "C++ should have no legal moves (checkmate)"
        assert len(py_moves) == 0, "python-chess should have no legal moves"
        assert py_board.is_checkmate()

    def test_stalemate_detection(self) -> None:
        """Test stalemate is detected correctly."""
        fen = "k7/2Q5/1K6/8/8/8/8/8 b - - 0 1"

        cpp_board = core.Board.from_fen(fen)
        py_board = pychess.Board(fen)

        cpp_moves = get_cpp_legal_moves_uci(cpp_board)
        py_moves = get_pychess_legal_moves_uci(py_board)

        assert cpp_moves == py_moves, (
            f"Legal moves mismatch in stalemate position!\n"
            f"C++: {sorted(cpp_moves)}\n"
            f"Py:  {sorted(py_moves)}"
        )
        assert len(cpp_moves) == 0, (
            f"C++ should have no legal moves (stalemate), got {cpp_moves}"
        )
        assert len(py_moves) == 0, "python-chess should have no legal moves"
        assert py_board.is_stalemate()

    def test_stalemate_white_trapped(self) -> None:
        """Test stalemate where white king is trapped by own pawn."""
        fen = "8/8/8/8/8/5k2/5p2/5K2 w - - 0 1"

        cpp_board = core.Board.from_fen(fen)
        py_board = pychess.Board(fen)

        cpp_moves = get_cpp_legal_moves_uci(cpp_board)
        py_moves = get_pychess_legal_moves_uci(py_board)

        assert cpp_moves == py_moves
        assert len(cpp_moves) == 0, f"C++ should have no legal moves, got {cpp_moves}"
        assert py_board.is_stalemate()
