import io
import os
import glob
import pytest
import chess.pgn
from engine._core import chess_engine_core as core

def get_pgn_files() -> list[str]:
    # Grab all .pgn files from data/raw/simple_games
    pgn_pattern = os.path.join("data", "raw", "simple_games", "*.pgn")
    return glob.glob(pgn_pattern)

@pytest.mark.parametrize("pgn_file", get_pgn_files())
def test_pgn_parity_with_python_chess(pgn_file: str) -> None:
    """
    Extensively test our C++ PGN parser against python-chess for parity.
    This guarantees that we handle headers, results, and SAN move extraction
    with exactly the same correctness as the reference library.
    """
    
    # 1. Parse using our C++ engine
    our_stream = core.pgn.PGNStream(pgn_file)
    our_games = list(our_stream)
    
    # 2. Parse using Python-chess
    pychess_games = []
    with open(pgn_file, "r") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            pychess_games.append(game)
            
    # Verify we extracted the exact same number of games
    assert len(our_games) == len(pychess_games), f"Game count mismatch in {pgn_file}"
    
    for i, (our_game, pychess_game) in enumerate(zip(our_games, pychess_games)):
        # -- A. Test Result Parity --
        # python-chess represents results via headers["Result"]
        pychess_result = pychess_game.headers.get("Result", "*")
        assert our_game.result == pychess_result, f"Game {i}: Result mismatch. Ours: {our_game.result}, PyChess: {pychess_result}"
        
        # -- B. Test Header Parity --
        # We ensure that every header we parsed matches exactly what pychess parsed.
        for key, our_val in our_game.headers.items():
            pychess_val = pychess_game.headers.get(key)
            assert our_val == pychess_val, f"Game {i}: Header '{key}' mismatch. Ours: {our_val}, PyChess: {pychess_val}"
            
        # -- C. Test Move (SAN) Extraction Parity --
        # pychess_game.mainline_moves() returns Move objects.
        # We need to format them back to SAN using a python-chess board to compare strings.
        pychess_board = pychess_game.board()
        pychess_sans = []
        for move in pychess_game.mainline_moves():
            pychess_sans.append(pychess_board.san(move))
            pychess_board.push(move)
            
        # Our engine extracts clean SANs natively
        assert len(our_game.moves) == len(pychess_sans), \
            f"Game {i}: Move count mismatch. Ours: {len(our_game.moves)}, PyChess: {len(pychess_sans)}\nOurs: {our_game.moves}\nPyChess: {pychess_sans}"
            
        for move_idx, (our_san, pychess_san) in enumerate(zip(our_game.moves, pychess_sans)):
            assert our_san == pychess_san, \
                f"Game {i}, Move {move_idx}: SAN mismatch. Ours: '{our_san}', PyChess: '{pychess_san}'"
