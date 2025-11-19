"""HalfKP representation utilities for neural network feature extraction."""

import torch

from engine._core import chess_engine_core as chess

# mypy: ignore-errors
# pyright: ignore
# pylint: skip-file
# ruff: noqa
# Constants for HalfKP representation
NUM_SQUARES = 64
NUM_PIECE_TYPES = 5
NUM_COLORS = 2
NUM_PLANES = NUM_SQUARES * NUM_PIECE_TYPES * NUM_COLORS + 1
HALFKP_FEATURES_PER_SIDE = NUM_SQUARES * NUM_PLANES


def orient_square(is_white_pov: bool, square: int) -> int:
    raise NotImplementedError()


def get_piece_index(piece: chess.Piece, is_white_pov: bool) -> int:
    raise NotImplementedError()


def halfkp_index(
    is_white_pov: bool, king_square: int, piece_square: int, piece: chess.Piece
) -> int:
    raise NotImplementedError()


def board_to_halfkp_indices(board: chess.Board, is_white_pov: bool) -> list[int]:
    raise NotImplementedError()


def board_to_halfkp_tensor(board: chess.Board) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError()


def get_halfkp_feature_size() -> int:
    return 2 * HALFKP_FEATURES_PER_SIDE


def board_to_input_tensor(board: chess.Board) -> torch.Tensor:
    raise NotImplementedError()


def create_accumulator_updates(
    board: chess.Board, move: chess.Move
) -> tuple[list[int], list[int]]:
    raise NotImplementedError()
