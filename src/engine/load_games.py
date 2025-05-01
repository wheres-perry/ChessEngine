"""
load_games.py

Utilities for locating PGN files in a directory and selecting a random chess position
from a PGN game using python-chess.
"""

import random
from pathlib import Path
from typing import Optional, List, Union
import chess
import chess.pgn


def get_all_files(directory: Union[str, Path]) -> List[Path]:
    """
    Return a list of all file paths in the given directory.

    Args:
        directory: Path or string pointing to a directory.

    Returns:
        List of Path objects for every file in the directory.
    """
    dir_path = Path(directory)
    return [p for p in dir_path.iterdir() if p.is_file()]


def random_file(games_dir: Union[str, Path] = Path("./games")) -> Optional[str]:
    """
    Choose a random filename from the specified games directory.

    Args:
        games_dir: Directory containing PGN files (default "./games").

    Returns:
        The selected file path as a string, or None if the directory is empty.
    """
    files = get_all_files(games_dir)
    if not files:
        return None
    return str(random.choice(files))


def random_board(pgnfile: Path) -> Optional[chess.Board]:
    """
    Parse the given PGN file, pick a random game, then pick a random position
    (including the starting position) from that game's mainline.

    Args:
        pgnfile: Path to a PGN file.

    Returns:
        A chess.Board at a random move in a random game, or None if no games
        were found.

    Raises:
        FileNotFoundError: If the PGN file does not exist.
        Exception: Propagates any other error encountered while reading.
    """
    games = []
    try:
        with open(pgnfile, 'r', encoding='utf-8', errors='ignore') as pgn_handle:
            while True:
                game = chess.pgn.read_game(pgn_handle)
                if game is None:
                    break  # End of file
                games.append(game)
    except FileNotFoundError:
        print(f"Error: PGN file not found at {pgnfile}")
        raise
    except Exception as e:
        print(f"An error occurred while reading the PGN file: {e}")
        raise

    if not games:
        return None

    game = random.choice(games)
    # Collect every node in the mainline (initial position included)
    mainline_nodes = [game]
    for node in game.mainline():
        mainline_nodes.append(node)

    random_node = random.choice(mainline_nodes)
    return random_node.board()
