"""Tests for CNN, HalfKP, and GNN feature extractors."""

import numpy as np
import pytest

from engine._core import chess_engine_core as core


def test_extract_cnn_starting_position() -> None:
    """Verify CNN feature extraction for the starting position."""
    board = core.Board()
    cnn = core.extractors.extract_cnn(board)

    assert isinstance(cnn, np.ndarray)
    assert cnn.shape == (17, 8, 8)
    assert cnn.dtype == np.float32

    # Channel 12: Side to move (White = 1.0)
    assert np.all(cnn[12] == 1.0)

    # Channels 13-16: Castling rights (All true at start)
    assert np.all(cnn[13] == 1.0)  # White Kingside
    assert np.all(cnn[14] == 1.0)  # White Queenside
    assert np.all(cnn[15] == 1.0)  # Black Kingside
    assert np.all(cnn[16] == 1.0)  # Black Queenside

    # Check some pieces (Channel 0: White Pawns)
    # Ranks 2 (index 1) should be all 1s
    assert np.all(cnn[0, 1, :] == 1.0)
    # Total white pawns should be 8
    assert np.sum(cnn[0]) == 8

    # Channel 6: Black Pawns
    # Ranks 7 (index 6) should be all 1s
    assert np.all(cnn[6, 6, :] == 1.0)
    assert np.sum(cnn[6]) == 8

    # Total pieces on board is 32
    assert np.sum(cnn[0:12]) == 32


def test_extract_cnn_black_to_move() -> None:
    """Verify CNN side-to-move channel for Black."""
    board = core.Board.from_fen(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )
    cnn = core.extractors.extract_cnn(board)

    # Channel 12: Side to move (Black = 0.0)
    assert np.all(cnn[12] == 0.0)


def test_extract_halfkp_starting_position() -> None:
    """Verify HalfKP feature extraction for the starting position."""
    board = core.Board()
    halfkp = core.extractors.extract_halfkp(board)

    assert isinstance(halfkp, np.ndarray)
    assert halfkp.dtype == np.int32
    # 32 pieces on board, kings are excluded from HalfKP features
    assert len(halfkp) == 30

    # Ensure all indices are within bounds
    # Max index is roughly 64 (king sq) * 10 (pieces) * 64 (piece sq) = 40960
    assert np.all(halfkp >= 0)
    assert np.all(halfkp < 40960)


def test_extract_halfkp_endgame() -> None:
    """Verify HalfKP extraction with minimal pieces."""
    board = core.Board.from_fen("8/8/8/8/8/8/4P3/K6k w - - 0 1")
    halfkp = core.extractors.extract_halfkp(board)

    # 1 pawn
    assert len(halfkp) == 1


def test_extract_gnn_starting_position() -> None:
    """Verify GNN graph extraction for the starting position."""
    board = core.Board()
    gnn = core.extractors.extract_gnn(board)

    assert isinstance(gnn, dict)
    assert "nodes" in gnn
    assert "edge_index" in gnn

    nodes = gnn["nodes"]
    edges = gnn["edge_index"]

    assert isinstance(nodes, np.ndarray)
    assert isinstance(edges, np.ndarray)

    assert nodes.dtype == np.int32
    assert edges.dtype == np.int32

    # 32 pieces on board -> 32 nodes
    assert nodes.shape == (32, 3)

    # Node attributes: square (0-63), piece_type (0-5), color (0-1)
    assert np.all((nodes[:, 0] >= 0) & (nodes[:, 0] < 64))
    assert np.all((nodes[:, 1] >= 0) & (nodes[:, 1] < 6))
    assert np.all((nodes[:, 2] >= 0) & (nodes[:, 2] < 2))

    # Edges shape should be (2, num_edges)
    assert len(edges.shape) == 2
    assert edges.shape[0] == 2

    num_edges = edges.shape[1]
    assert num_edges > 0

    # Ensure all edge indices are valid node indices
    assert np.all(edges >= 0)
    assert np.all(edges < 32)


def test_extract_gnn_empty_board() -> None:
    """Verify GNN extraction with minimal pieces (just kings)."""
    board = core.Board.from_fen("8/8/8/8/8/8/8/K6k w - - 0 1")
    gnn = core.extractors.extract_gnn(board)

    nodes = gnn["nodes"]
    edges = gnn["edge_index"]

    assert nodes.shape == (2, 3)
    # Kings don't attack each other here, so no edges
    assert edges.shape[1] == 0
