import chess
import pytest

from src.engine.constants import *
from src.engine.evaluators.eval import Eval
from src.engine.evaluators.simple_eval import SimpleEval
# from src.engine.evaluators.simple_nn_eval import NN_Eval, ChessNN


# def create_nn_eval(board):
#     model = ChessNN()
#     return NN_Eval(board, model_instance=model)


@pytest.fixture
def evaluator_fixture(request):
    param = request.param
    if isinstance(param, tuple):
        board, evaluator = param
    else:
        board = chess.Board()
        evaluator = param
    # if evaluator == NN_Eval:
    #     evaluator = create_nn_eval
    if board is None:
        return lambda b: evaluator(b)
    return evaluator(board)


@pytest.mark.parametrize(
    "evaluator_fixture",
    [
        (chess.Board(), SimpleEval),
        # (chess.Board(), NN_Eval),
    ],
    indirect=True,
)
def test_starting_position(evaluator_fixture):
    score = evaluator_fixture.evaluate()
    assert score == 0.0


@pytest.mark.parametrize(
    "evaluator_fixture",
    [
        (None, SimpleEval),
        # (None, NN_Eval),
    ],
    indirect=True,
)
def test_white_checkmate(evaluator_fixture):
    eval_factory = evaluator_fixture
    board = chess.Board()
    moves = ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]
    for move_san in moves:
        board.push_san(move_san)
    evaluator_instance = eval_factory(board)
    score = evaluator_instance.evaluate()
    assert board.is_checkmate()
    assert score == float("inf")


@pytest.mark.parametrize(
    "evaluator_fixture",
    [
        (None, SimpleEval),
        # (None, NN_Eval),
    ],
    indirect=True,
)
def test_black_checkmate(evaluator_fixture):
    eval_factory = evaluator_fixture
    board = chess.Board()
    moves = ["f3", "e5", "g4", "Qh4#"]
    for move_san in moves:
        board.push_san(move_san)
    evaluator_instance = eval_factory(board)
    score = evaluator_instance.evaluate()
    assert board.is_checkmate()
    assert score == float("-inf")


@pytest.mark.parametrize(
    "evaluator_fixture",
    [
        (chess.Board("7k/8/8/8/8/8/2q5/K7 w - - 0 1"), SimpleEval),
        # (chess.Board("7k/8/8/8/8/8/2q5/K7 w - - 0 1"), NN_Eval),
    ],
    indirect=True,
)
def test_stalemate(evaluator_fixture):
    board = evaluator_fixture.board
    assert board.is_stalemate()
    score = evaluator_fixture.evaluate()
    assert score == 0.0


@pytest.mark.parametrize(
    "evaluator_fixture",
    [
        (chess.Board("k7/8/8/8/8/8/R7/K7 w - - 0 1"), SimpleEval),
        # (chess.Board("k7/8/8/8/8/8/R7/K7 w - - 0 1"), NN_Eval),
    ],
    indirect=True,
)
def test_material_advantage_white(evaluator_fixture):
    score = evaluator_fixture.evaluate()
    expected_score = PIECE_VALUES[chess.ROOK]
    assert score == expected_score


@pytest.mark.parametrize(
    "evaluator_fixture",
    [
        (chess.Board("k1rb4/8/8/8/8/8/8/K7 w - - 0 1"), SimpleEval),
        # (chess.Board("k1rb4/8/8/8/8/8/8/K7 w - - 0 1"), NN_Eval),
    ],
    indirect=True,
)
def test_material_advantage_black_multiple_pieces(evaluator_fixture):
    score = evaluator_fixture.evaluate()
    expected_score = -(PIECE_VALUES[chess.ROOK] + PIECE_VALUES[chess.BISHOP])
    assert score == expected_score


@pytest.mark.parametrize(
    "evaluator_fixture",
    [
        (chess.Board("k7/8/K7/8/8/8/8/8 w - - 0 1"), SimpleEval),
        # (chess.Board("k7/8/K7/8/8/8/8/8 w - - 0 1"), NN_Eval),
    ],
    indirect=True,
)
def test_kings_only(evaluator_fixture):
    score = evaluator_fixture.evaluate()
    assert score == 0.0
