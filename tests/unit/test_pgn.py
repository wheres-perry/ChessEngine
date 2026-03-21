import pytest
from engine._core import chess_engine_core as core

def test_pgn_parser() -> None:
    # Use the test PGN we just created
    pgn_path = "data/raw/simple_games/test_parser.pgn"
    
    stream = core.pgn.PGNStream(pgn_path)
    games = list(stream)
    
    assert len(games) == 2
    
    # First game assertions
    game1 = games[0]
    assert game1.headers["White"] == "Carlsen,M"
    assert game1.headers["Black"] == "Bu Xiangzhi"
    assert game1.result == "1-0"
    
    # Verify move cleaning (Should not contain '{', '}', '(', ')', ';', '$1', or '1.')
    assert "e4" in game1.moves
    assert "Nf3" in game1.moves
    assert "Nd5" in game1.moves
    
    # Check for stripped elements
    assert "1." not in game1.moves
    assert "8..." not in game1.moves
    assert "9." not in game1.moves
    assert "$1" not in game1.moves
    assert "This" not in game1.moves # Part of comment
    assert "is" not in game1.moves
    assert "9." not in game1.moves # Inside RAV
    
    # Total moves should be clean SANs
    # E.g. [e4, e5, Nf3, Nc6, Bb5, a6, Ba4, Nf6, d3, b5, Bb3, Bc5, Nc3, d6, Nd5, Na5, Nxf6+, gxf6, O-O, Nxb3, axb3, Rg8, Kh1, f5, Nh4, f4, Nf5, Qg5, Qf3, Be6, d4, Bb6, c3, Kd7, b4, a5, bxa5, Rxa5, Bd2, Rxa1, Rxa1, h5, h3, Qg6, Rg1, f6, Qd3, c6, Qxb5]
    assert game1.moves[-1] == "Qxb5"
    assert len(game1.moves) == 49

    # Second game assertions
    game2 = games[1]
    assert game2.headers["Event"] == "Short Game"
    assert game2.result == "0-1"
    assert game2.moves == ["f3", "e5", "g4", "Qh4#"]


def test_pgn_integration_with_board() -> None:
    # Ensure that parsed SANs can be fed directly to the Board
    pgn_path = "data/raw/simple_games/test_parser.pgn"
    stream = core.pgn.PGNStream(pgn_path)
    game2 = list(stream)[1]
    
    board = core.Board()
    for san in game2.moves:
        move = board.push_san(san)
        assert move is not None
        
    assert board.is_check()
    assert board.is_game_over() == core.GameState.CHECKMATE