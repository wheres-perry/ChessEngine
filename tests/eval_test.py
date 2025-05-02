import os
import chess
import pytest
from dotenv import load_dotenv
from stockfish import Stockfish
from src.engine.eval import SimpleEval
from src.engine.load_games import random_board

load_dotenv()
stockfish_path_from_env = os.getenv("STOCKFISH_PATH")

@pytest.fixture
def stockfish_engine():
    if not stockfish_path_from_env:
         raise ValueError("STOCKFISH_PATH not found in environment variables (.env).")
    if not os.path.exists(stockfish_path_from_env):
         raise FileNotFoundError(f"Stockfish executable not found at specified path: {stockfish_path_from_env}.")

    try:
        engine = Stockfish(path=stockfish_path_from_env, depth=1, parameters={"Threads": 1, "Hash": 128})
        engine.get_parameters()
    except Exception as e:
         raise RuntimeError(f"Stockfish engine failed to initialize or respond: {e}") from e

    return engine

@pytest.fixture
def random_chess_board() -> chess.Board:
    return random_board()

@pytest.mark.parametrize("i", range(5))
def test_basic_vs_stockfish(random_chess_board: chess.Board, stockfish_engine: Stockfish, i: int):
    board = random_chess_board

    if board.is_game_over(claim_draw=True):
        pytest.skip(f"Skipping game over position: {board.fen()}")

    simple_evaluator = SimpleEval(board)
    basic_score = simple_evaluator.basic_evaluate()

    stockfish_engine.set_fen_position(board.fen())
    eval_dict = stockfish_engine.get_evaluation()

    assert eval_dict is not None, f"Stockfish returned None evaluation for FEN: {board.fen()}"

    stockfish_score = 0.0

    if eval_dict["type"] == "cp":
        stockfish_score = eval_dict["value"] / 100.0
    elif eval_dict["type"] == "mate":
        mate_val = eval_dict["value"]
        if mate_val > 0:
             stockfish_score = float('inf')
        elif mate_val < 0:
             stockfish_score = float('-inf')
        else:
             if board.turn == chess.WHITE:
                 stockfish_score = float('-inf')
             else:
                 stockfish_score = float('inf')

    tolerance = 2.0

    print(f"\n--- Test Run {i+1} ---")
    print(f"FEN: {board.fen()}")
    print(f"SimpleEval Score: {basic_score}")
    print(f"Stockfish Score ({eval_dict['type']}): {stockfish_score} (Raw: {eval_dict['value']})")


    if basic_score == float('inf'):
        assert stockfish_score == float('inf'), f"SimpleEval: inf, Stockfish: {eval_dict}"
    elif basic_score == float('-inf'):
        assert stockfish_score == float('-inf'), f"SimpleEval: -inf, Stockfish: {eval_dict}"
    elif stockfish_score == float('inf'):
         assert basic_score == float('inf'), f"SimpleEval: {basic_score}, Stockfish: mate inf"
    elif stockfish_score == float('-inf'):
         assert basic_score == float('-inf'), f"SimpleEval: {basic_score}, Stockfish: mate -inf"
    else:
        assert isinstance(basic_score, (int, float)) and basic_score != float('inf') and basic_score != float('-inf'), f"Basic score is not a finite number: {basic_score}"
        assert isinstance(stockfish_score, (int, float)) and stockfish_score != float('inf') and stockfish_score != float('-inf'), f"Stockfish score is not a finite number: {stockfish_score}"
        assert abs(basic_score - stockfish_score) < tolerance, \
            f"Score difference |{basic_score:.2f} - {stockfish_score:.2f}| = {abs(basic_score - stockfish_score):.2f} > tolerance {tolerance}"

import os
import chess
import pytest
from dotenv import load_dotenv
from stockfish import Stockfish
from src.engine.eval import SimpleEval
from src.engine.load_games import random_board

load_dotenv()
stockfish_path_from_env = os.getenv("STOCKFISH_PATH")

@pytest.fixture
def stockfish_engine():
    if not stockfish_path_from_env:
         raise ValueError("STOCKFISH_PATH not found in environment variables (.env).")
    if not os.path.exists(stockfish_path_from_env):
         raise FileNotFoundError(f"Stockfish executable not found at specified path: {stockfish_path_from_env}.")

    try:
        engine = Stockfish(path=stockfish_path_from_env, depth=1, parameters={"Threads": 1, "Hash": 128})
        engine.get_parameters()
    except Exception as e:
         raise RuntimeError(f"Stockfish engine failed to initialize or respond: {e}") from e

    return engine

@pytest.fixture
def random_chess_board() -> chess.Board:
    return random_board()

@pytest.mark.parametrize("i", range(5))
def test_basic_vs_stockfish(random_chess_board: chess.Board, stockfish_engine: Stockfish, i: int):
    board = random_chess_board

    if board.is_game_over(claim_draw=True):
        pytest.skip(f"Skipping game over position: {board.fen()}")

    simple_evaluator = SimpleEval(board)
    basic_score = simple_evaluator.basic_evaluate()

    stockfish_engine.set_fen_position(board.fen())
    eval_dict = stockfish_engine.get_evaluation()

    assert eval_dict is not None, f"Stockfish returned None evaluation for FEN: {board.fen()}"

    stockfish_score = 0.0

    if eval_dict["type"] == "cp":
        stockfish_score = eval_dict["value"] / 100.0
    elif eval_dict["type"] == "mate":
        mate_val = eval_dict["value"]
        if mate_val > 0:
             stockfish_score = float('inf')
        elif mate_val < 0:
             stockfish_score = float('-inf')
        else:
             if board.turn == chess.WHITE:
                 stockfish_score = float('-inf')
             else:
                 stockfish_score = float('inf')

    tolerance = 2.0

    print(f"\n--- Test Run {i+1} ---")
    print(f"FEN: {board.fen()}")
    print(f"SimpleEval Score: {basic_score}")
    print(f"Stockfish Score ({eval_dict['type']}): {stockfish_score} (Raw: {eval_dict['value']})")


    if basic_score == float('inf'):
        assert stockfish_score == float('inf'), f"SimpleEval: inf, Stockfish: {eval_dict}"
    elif basic_score == float('-inf'):
        assert stockfish_score == float('-inf'), f"SimpleEval: -inf, Stockfish: {eval_dict}"
    elif stockfish_score == float('inf'):
         assert basic_score == float('inf'), f"SimpleEval: {basic_score}, Stockfish: mate inf"
    elif stockfish_score == float('-inf'):
         assert basic_score == float('-inf'), f"SimpleEval: {basic_score}, Stockfish: mate -inf"
    else:
        assert isinstance(basic_score, (int, float)) and basic_score != float('inf') and basic_score != float('-inf'), f"Basic score is not a finite number: {basic_score}"
        assert isinstance(stockfish_score, (int, float)) and stockfish_score != float('inf') and stockfish_score != float('-inf'), f"Stockfish score is not a finite number: {stockfish_score}"
        assert abs(basic_score - stockfish_score) < tolerance, \
            f"Score difference |{basic_score:.2f} - {stockfish_score:.2f}| = {abs(basic_score - stockfish_score):.2f} > tolerance {tolerance}"

