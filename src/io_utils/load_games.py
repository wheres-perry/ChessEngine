import random
from pathlib import Path

import chess
import chess.pgn


def get_all_files(directory: str | Path) -> list[Path]:
    dir_path = Path(directory)
    return [p for p in dir_path.iterdir() if p.is_file()]


def random_file(games_dir: str | Path = Path("./data/raw/simple_games")) -> Path:
    files = get_all_files(games_dir)
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
        with open(pgnfile, "r", encoding="utf-8", errors="ignore") as pgn_handle:
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
