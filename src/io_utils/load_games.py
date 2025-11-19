"""Utility helpers for sampling random positions from PGN archives."""

import logging
import random
from pathlib import Path

import chess.pgn as chess_pgn

from engine._core import chess_engine_core as chess

logger = logging.getLogger(__name__)


def get_all_files(directory: str | Path) -> list[Path]:
    """Get all files in a directory.

    Args:
        directory: Directory path to search.

    Returns:
        List of Path objects for all files in the directory.
    """
    dir_path = Path(directory)
    return [p for p in dir_path.iterdir() if p.is_file()]


def random_file(games_dir: str | Path = Path("./data/raw/simple_games")) -> Path | None:
    """Select a random file from the games directory.

    Args:
        games_dir: Directory containing game files.

    Returns:
        Random file path, or None if directory is empty.
    """
    files = get_all_files(games_dir)
    if not files:
        return None
    return random.choice(files)


def random_board(
    games_dir: str | Path = Path("./data/raw/simple_games"),
) -> chess.Board | None:
    """Load a random board position from PGN files in the directory.

    Args:
        games_dir: Directory containing PGN game files.

    Returns:
        Random chess board position, or None if no games found.

    Raises:
        FileNotFoundError: If PGN file cannot be found.
        Exception: If error occurs reading PGN file.
    """
    pgnfile = random_file(games_dir)
    if pgnfile is None:
        logger.warning("No PGN files found in the specified directory.")
        return None
    games = []
    try:
        with open(pgnfile, encoding="utf-8", errors="ignore") as pgn_handle:
            while True:
                game = chess_pgn.read_game(pgn_handle)
                if game is None:
                    break  # End of file
                games.append(game)
    except FileNotFoundError:
        logger.error("PGN file not found at %s", pgnfile)
        raise
    except Exception as e:
        logger.exception("Error while reading PGN file %s", pgnfile)
        raise
    if not games:
        return None
    game = random.choice(games)
    # Collect every node in the mainline (initial position included)

    mainline_nodes = list(game.mainline())
    # Include the initial position by choosing between game root and mainline nodes
    if mainline_nodes:
        # Choose randomly between initial position and any move position
        if random.choice([True, False]):
            return chess.Board.from_fen(game.board().fen())
        random_node = random.choice(mainline_nodes)
        return chess.Board.from_fen(random_node.board().fen())
    # Game has no moves, return initial position
    return chess.Board.from_fen(game.board().fen())
