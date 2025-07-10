import random
from pathlib import Path

import chess
import chess.pgn


def get_all_files(directory: str | Path) -> list[Path]:
    dir_path = Path(directory)
    return [p for p in dir_path.iterdir() if p.is_file()]


def random_file(games_dir: str | Path = Path("./data/raw/simple_games")) -> Path | None:
    files = get_all_files(games_dir)
    if not files:
        return None
    return random.choice(files)


def random_board(
    games_dir: str | Path = Path("./data/raw/simple_games"),
) -> chess.Board | None:
    pgnfile = random_file(games_dir)
    if pgnfile is None:
        print("No PGN files found in the specified directory.")
        return None
    games = []
    try:
        with open(pgnfile, encoding="utf-8", errors="ignore") as pgn_handle:
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

    mainline_nodes = list(game.mainline())
    # Include the initial position by choosing between game root and mainline nodes
    if mainline_nodes:
        # Choose randomly between initial position and any move position
        if random.choice([True, False]):
            return game.board()  # Initial position
        random_node = random.choice(mainline_nodes)
        return random_node.board()
    # Game has no moves, return initial position
    return game.board()
