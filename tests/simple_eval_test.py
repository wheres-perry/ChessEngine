import chess
import pytest
from src.engine.constants import *
from engine.evaluators.simple_eval import SimpleEval


def test_starting_position():
    board = chess.Board()
    evaluator = SimpleEval(board)
    score = evaluator.basic_evaluate()
    assert score == 0.0


def test_white_checkmate():
    board = chess.Board()
    moves = ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]
    for move_san in moves:
        board.push_san(move_san)
    evaluator = SimpleEval(board)
    score = evaluator.basic_evaluate()
    assert board.is_checkmate()
    assert score == float('inf')


def test_black_checkmate():
    board = chess.Board()
    moves = ["f3", "e5", "g4", "Qh4#"]
    for move_san in moves:
        board.push_san(move_san)
    evaluator = SimpleEval(board)
    score = evaluator.basic_evaluate()
    assert board.is_checkmate()
    assert score == float('-inf')


def test_stalemate():
    fen_stalemate = "7k/8/8/8/8/8/2q5/K7 w - - 0 1"  # Corrected FEN for stalemate
    board = chess.Board(fen_stalemate)
    assert board.is_stalemate()
    evaluator = SimpleEval(board)
    score = evaluator.basic_evaluate()
    assert score == 0.0


def test_material_advantage_white():
    board = chess.Board("k7/8/8/8/8/8/R7/K7 w - - 0 1")
    evaluator = SimpleEval(board)
    score = evaluator.basic_evaluate()
    expected_score = PIECE_VALUES[chess.ROOK]
    assert score == expected_score


def test_complex_material_advantage():
    board = chess.Board("8/8/8/8/8/k7/B7/K1Q5 w - - 0 1")
    evaluator = SimpleEval(board)

    expected_score = PIECE_VALUES[chess.QUEEN] + PIECE_VALUES[chess.BISHOP]

    score = evaluator.evaluate()
    assert score == expected_score


def test_kings_only():
    board = chess.Board("k7/8/K7/8/8/8/8/8 w - - 0 1")
    evaluator = SimpleEval(board)
    score = evaluator.basic_evaluate()
    assert score == 0.0
