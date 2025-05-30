from pathlib import Path

import chess
import chess.pgn
import pytest
from src.io.load_games import get_all_files, random_board, random_file


@pytest.fixture
def temp_games_dir(tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    return games_dir


@pytest.fixture
def populated_games_dir(temp_games_dir):
    pgn_content_1 = """







[Event "Test Game 1"]







[Site "?"]







[Date "?"]







[Round "?"]







[White "?"]







[Black "?"]







[Result "*"]















1. e4 e5 2. Nf3 Nc6 *







"""
    pgn_file_1 = temp_games_dir / "game1.pgn"
    pgn_file_1.write_text(pgn_content_1)

    pgn_content_2 = """







[Event "Test Game 2"]







1. d4 d5 *







"""
    pgn_file_2 = temp_games_dir / "game2.pgn"
    pgn_file_2.write_text(pgn_content_2)

    non_pgn_file = temp_games_dir / "notes.txt"
    non_pgn_file.write_text("Some notes.")
    return temp_games_dir


@pytest.fixture
def empty_pgn_file_dir(temp_games_dir):
    empty_pgn = temp_games_dir / "empty.pgn"
    empty_pgn.write_text("")
    return temp_games_dir


@pytest.fixture
def no_games_pgn_file_dir(temp_games_dir):
    no_games_pgn = temp_games_dir / "no_games.pgn"
    no_games_pgn.write_text('[Event "Only Headers"]\n[Result "*"]\n')
    return temp_games_dir


class TestGetAllFiles:
    def test_get_all_files_populated(self, populated_games_dir):
        files = get_all_files(populated_games_dir)
        assert len(files) == 3
        fnames = sorted([f.name for f in files])
        assert fnames == ["game1.pgn", "game2.pgn", "notes.txt"]

    def test_get_all_files_empty(self, temp_games_dir):
        files = get_all_files(temp_games_dir)
        assert len(files) == 0

    def test_get_all_files_non_existent_dir(self):
        non_existent_path = Path("./non_existent_test_dir_56789")
        with pytest.raises(FileNotFoundError):
            get_all_files(non_existent_path)

    def test_get_all_files_with_file_path(self, populated_games_dir):
        file_path = populated_games_dir / "game1.pgn"
        with pytest.raises(NotADirectoryError):
            get_all_files(file_path)


class TestRandomFile:
    def test_random_file_populated(self, populated_games_dir):
        chosen_file = random_file(populated_games_dir)
        assert isinstance(chosen_file, Path)
        assert chosen_file.name in ["game1.pgn", "game2.pgn", "notes.txt"]
        assert chosen_file.exists()

    def test_random_file_empty_dir(self, temp_games_dir):
        with pytest.raises(IndexError):
            random_file(temp_games_dir)

    def test_random_file_non_existent_dir(self):
        non_existent_path = Path("./non_existent_test_dir_12345")
        with pytest.raises(FileNotFoundError):
            random_file(non_existent_path)


class TestRandomBoard:
    def test_random_board_populated(self, populated_games_dir):
        board = random_board(populated_games_dir)
        assert isinstance(board, chess.Board)

    def test_random_board_empty_pgn(self, empty_pgn_file_dir):
        board = random_board(empty_pgn_file_dir)
        assert board is None

    def test_random_board_multi_game_pgn(self, multi_game_pgn_dir):
        board = random_board(multi_game_pgn_dir)
        assert isinstance(board, chess.Board)

    def test_random_board_empty_dir(self, temp_games_dir):
        with pytest.raises(IndexError):
            random_board(temp_games_dir)

    def test_random_board_non_existent_dir(self):
        non_existent_path = Path("./non_existent_test_dir_ABCDE")
        with pytest.raises(FileNotFoundError):
            random_board(non_existent_path)
