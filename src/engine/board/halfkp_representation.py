"""HalfKP representation utilities for neural network feature extraction.

This module provides both Python fallback and optimized C++ implementations
for HalfKP (Half King-Piece) feature extraction used in NNUE-style neural
network chess evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from engine._core import chess_engine_core as chess

if TYPE_CHECKING:
    from engine._core.chess_engine_core import Board, Move, Piece

# Constants for HalfKP representation
NUM_SQUARES = 64
NUM_PIECE_TYPES = 5  # Exclude king
NUM_COLORS = 2
NUM_PLANES = NUM_SQUARES * NUM_PIECE_TYPES * NUM_COLORS + 1
HALFKP_FEATURES_PER_SIDE = NUM_SQUARES * NUM_PLANES
TOTAL_FEATURES = 2 * HALFKP_FEATURES_PER_SIDE


def orient_square(is_white_pov: bool, square: int) -> int:
    """Orient square from perspective of given color.

    Args:
        is_white_pov: True for white's perspective, False for black's
        square: Square index (0-63)

    Returns:
        Oriented square index
    """
    try:
        return int(chess.halfkp.orient_square(is_white_pov, square))
    except AttributeError:
        # Fallback if C++ not available
        return square if is_white_pov else (square ^ 56)


def get_piece_index(piece: Piece, is_white_pov: bool) -> int:
    """Get piece index (0-9) for HalfKP encoding.

    Friendly pieces: 0-4 (P,N,B,R,Q), opponent pieces: 5-9

    Args:
        piece: Chess piece
        is_white_pov: True for white's perspective

    Returns:
        Piece index for HalfKP encoding
    """
    try:
        return int(
            chess.halfkp.get_piece_index(piece.piece_type, piece.color, is_white_pov)
        )
    except AttributeError:
        # Fallback
        base_idx = int(piece.piece_type)
        is_friendly = (is_white_pov and piece.color == chess.WHITE) or (
            not is_white_pov and piece.color == chess.BLACK
        )
        return base_idx if is_friendly else (base_idx + NUM_PIECE_TYPES)


def halfkp_index(
    is_white_pov: bool,
    king_square: int,
    piece_square: int,
    piece: Piece,
) -> int:
    """Compute HalfKP feature index for a single piece.

    Args:
        is_white_pov: True for white's perspective
        king_square: King square index
        piece_square: Piece square index
        piece: Chess piece

    Returns:
        HalfKP feature index
    """
    try:
        return int(
            chess.halfkp.halfkp_index(
                is_white_pov,
                king_square,
                piece_square,
                piece.piece_type,
                piece.color,
            )
        )
    except AttributeError:
        # Fallback
        oriented_king = orient_square(is_white_pov, king_square)
        oriented_piece = orient_square(is_white_pov, piece_square)
        piece_idx = get_piece_index(piece, is_white_pov)
        return oriented_king * NUM_PLANES + piece_idx * NUM_SQUARES + oriented_piece


def board_to_halfkp_indices(board: Board, is_white_pov: bool) -> list[int]:
    """Extract all active HalfKP feature indices for one perspective.

    Args:
        board: Chess board
        is_white_pov: True for white's perspective

    Returns:
        List of active feature indices (sparse representation)
    """
    try:
        return list(chess.halfkp.board_to_halfkp_indices(board, is_white_pov))
    except AttributeError:
        # Fallback: Python implementation
        indices: list[int] = []
        pov_color = chess.WHITE if is_white_pov else chess.BLACK
        king_square = board.king(pov_color)
        if king_square is None:
            return indices

        # Iterate over all piece types (excluding king)
        for pt in range(NUM_PIECE_TYPES):
            for color in [chess.WHITE, chess.BLACK]:
                squares = board.pieces(pt, color)
                for sq in squares:
                    piece = board.piece_at(sq)
                    if piece:
                        idx = halfkp_index(is_white_pov, king_square, sq, piece)
                        indices.append(idx)

        return indices


def board_to_halfkp_tensor(board: Board) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert board to HalfKP tensors for both perspectives.

    Args:
        board: Chess board

    Returns:
        Tuple of (white_tensor, black_tensor) with shape (HALFKP_FEATURES_PER_SIDE,)
    """
    # White perspective
    white_indices = board_to_halfkp_indices(board, is_white_pov=True)
    white_tensor = torch.zeros(HALFKP_FEATURES_PER_SIDE, dtype=torch.float32)
    for idx in white_indices:
        white_tensor[idx] = 1.0
    white_tensor[HALFKP_FEATURES_PER_SIDE - 1] = 1.0  # Bias plane

    # Black perspective
    black_indices = board_to_halfkp_indices(board, is_white_pov=False)
    black_tensor = torch.zeros(HALFKP_FEATURES_PER_SIDE, dtype=torch.float32)
    for idx in black_indices:
        black_tensor[idx] = 1.0
    black_tensor[HALFKP_FEATURES_PER_SIDE - 1] = 1.0  # Bias plane

    return white_tensor, black_tensor


def get_halfkp_feature_size() -> int:
    """Get total HalfKP feature size (both perspectives).

    Returns:
        Total feature size
    """
    return TOTAL_FEATURES


def board_to_input_tensor(board: Board) -> torch.Tensor:
    """Convert board to dense input tensor for neural network.

    Args:
        board: Chess board

    Returns:
        Tensor of shape (TOTAL_FEATURES,) with both perspectives concatenated
    """
    try:
        # Use C++ implementation for maximum performance
        features = chess.halfkp.board_to_input_tensor(board)
        return torch.tensor(features, dtype=torch.float32)
    except AttributeError:
        # Fallback: use Python implementation
        white_tensor, black_tensor = board_to_halfkp_tensor(board)
        return torch.cat([white_tensor, black_tensor])


def create_accumulator_updates(
    board: Board, move: Move
) -> tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]:
    """Compute incremental updates for a move (both perspectives).

    Args:
        board: Chess board before the move
        move: Move to apply

    Returns:
        Tuple of ((white_added, white_removed), (black_added, black_removed))
    """
    try:
        white_update, black_update = chess.halfkp.create_accumulator_updates(
            board, move
        )
        return (
            (white_update.added_indices, white_update.removed_indices),
            (black_update.added_indices, black_update.removed_indices),
        )
    except AttributeError as exc:
        # Fallback: not implemented in Python (requires full recomputation)
        raise NotImplementedError(
            "Accumulator updates require C++ implementation for performance"
        ) from exc
