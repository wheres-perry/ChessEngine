import random

import chess.pgn

# mypy: ignore-errors
# pyright: ignore
# pylint: skip-file
# ruff: noqa
input_pgn_path = "benchmarks/lichess_db_standard_rated_2014-07.pgn/lichess_db_standard_rated_2014-07.pgn"
output_fens_path = "./fens.txt"
n_fens = 1_000_000


def random_fen_sampling(pgn_path, output_path, fen_target, prob=0.1):
    count = 0
    with (
        open(pgn_path, encoding="utf-8") as pgn_file,
        open(output_path, "w", encoding="utf-8") as out_file,
    ):
        # Go through each game
        while count < fen_target:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                # Reached end, optionally rewind or break
                pgn_file.seek(0)
                continue
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if random.random() < prob:
                    out_file.write(board.fen() + "\n")
                    count += 1
                    if count % 10000 == 0:
                        print(f"Collected {count} FENs...")
                    if count >= fen_target:
                        break


if __name__ == "__main__":
    random_fen_sampling(input_pgn_path, output_fens_path, n_fens)
    print("Done!")
