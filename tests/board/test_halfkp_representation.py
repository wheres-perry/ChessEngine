"""Tests for HalfKP feature extraction."""

from __future__ import annotations

import pytest
import torch

from engine._core import chess_engine_core as chess
from src.engine.board.halfkp_representation import (
    HALFKP_FEATURES_PER_SIDE,
    NUM_PIECE_TYPES,
    NUM_SQUARES,
    TOTAL_FEATURES,
    board_to_halfkp_indices,
    board_to_halfkp_tensor,
    board_to_input_tensor,
    get_halfkp_feature_size,
    get_piece_index,
    halfkp_index,
    orient_square,
)


class TestHalfKPConstants:
    """Test HalfKP constants are correct."""

    def test_feature_dimensions(self) -> None:
        """Test that feature dimensions are calculated correctly."""
        assert NUM_SQUARES == 64
        assert NUM_PIECE_TYPES == 5  # Exclude king
        # NUM_PLANES = 64 * 5 * 2 + 1 = 641
        expected_planes = NUM_SQUARES * NUM_PIECE_TYPES * 2 + 1
        assert expected_planes == 641
        # HALFKP_FEATURES_PER_SIDE = 64 * 641 = 41,024
        assert HALFKP_FEATURES_PER_SIDE == 41024
        # TOTAL_FEATURES = 2 * 41,024 = 82,048
        assert TOTAL_FEATURES == 82048

    def test_get_feature_size(self) -> None:
        """Test get_halfkp_feature_size returns correct value."""
        assert get_halfkp_feature_size() == TOTAL_FEATURES


class TestOrientSquare:
    """Test square orientation from different perspectives."""

    def test_white_perspective_no_change(self) -> None:
        """Test that white perspective doesn't change squares."""
        for square in range(64):
            assert orient_square(is_white_pov=True, square=square) == square

    def test_black_perspective_vertical_flip(self) -> None:
        """Test that black perspective flips squares vertically."""
        # a1 (0) -> a8 (56)
        assert orient_square(is_white_pov=False, square=0) == 56
        # h1 (7) -> h8 (63)
        assert orient_square(is_white_pov=False, square=7) == 63
        # a8 (56) -> a1 (0)
        assert orient_square(is_white_pov=False, square=56) == 0
        # h8 (63) -> h1 (7)
        assert orient_square(is_white_pov=False, square=63) == 7
        # e4 (28) -> e5 (36)
        assert orient_square(is_white_pov=False, square=28) == 36


class TestGetPieceIndex:
    """Test piece index calculation for HalfKP encoding."""

    def test_white_pov_friendly_pieces(self) -> None:
        """Test that white pieces get indices 0-4 from white's perspective."""
        board = chess.Board()
        # Get a white pawn
        pawn = board.piece_at(8)  # a2
        assert pawn is not None
        assert get_piece_index(pawn, is_white_pov=True) == 0  # Pawn

    def test_white_pov_opponent_pieces(self) -> None:
        """Test that black pieces get indices 5-9 from white's perspective."""
        board = chess.Board()
        # Get a black pawn
        pawn = board.piece_at(48)  # a7
        assert pawn is not None
        assert get_piece_index(pawn, is_white_pov=True) == 5  # Opponent pawn

    def test_black_pov_friendly_pieces(self) -> None:
        """Test that black pieces get indices 0-4 from black's perspective."""
        board = chess.Board()
        # Get a black pawn
        pawn = board.piece_at(48)  # a7
        assert pawn is not None
        assert get_piece_index(pawn, is_white_pov=False) == 0  # Pawn

    def test_black_pov_opponent_pieces(self) -> None:
        """Test that white pieces get indices 5-9 from black's perspective."""
        board = chess.Board()
        # Get a white pawn
        pawn = board.piece_at(8)  # a2
        assert pawn is not None
        assert get_piece_index(pawn, is_white_pov=False) == 5  # Opponent pawn


class TestHalfKPIndex:
    """Test HalfKP feature index calculation."""

    def test_index_calculation(self) -> None:
        """Test that HalfKP index is calculated correctly."""
        board = chess.Board()
        # White king at e1 (4), white pawn at e2 (12)
        king_square = 4
        pawn_square = 12
        pawn = board.piece_at(pawn_square)
        assert pawn is not None

        idx = halfkp_index(
            is_white_pov=True,
            king_square=king_square,
            piece_square=pawn_square,
            piece=pawn,
        )

        # Index should be deterministic and within valid range
        assert 0 <= idx < HALFKP_FEATURES_PER_SIDE

    def test_different_perspectives_different_indices(self) -> None:
        """Test that same position gives different indices from perspectives."""
        board = chess.Board()
        king_square = 4  # e1
        pawn_square = 12  # e2
        pawn = board.piece_at(pawn_square)
        assert pawn is not None

        white_idx = halfkp_index(
            is_white_pov=True,
            king_square=king_square,
            piece_square=pawn_square,
            piece=pawn,
        )

        black_idx = halfkp_index(
            is_white_pov=False,
            king_square=king_square,
            piece_square=pawn_square,
            piece=pawn,
        )

        # Different perspectives should give different indices
        assert white_idx != black_idx


class TestBoardToHalfKPIndices:
    """Test sparse HalfKP feature extraction."""

    def test_starting_position_has_correct_count(self) -> None:
        """Test that starting position has 30 active features per side."""
        board = chess.Board()

        white_indices = board_to_halfkp_indices(board, is_white_pov=True)
        black_indices = board_to_halfkp_indices(board, is_white_pov=False)

        # 32 pieces - 2 kings = 30 pieces per perspective
        assert len(white_indices) == 30
        assert len(black_indices) == 30

    def test_indices_are_unique(self) -> None:
        """Test that all indices are unique."""
        board = chess.Board()

        white_indices = board_to_halfkp_indices(board, is_white_pov=True)
        assert len(white_indices) == len(set(white_indices))

        black_indices = board_to_halfkp_indices(board, is_white_pov=False)
        assert len(black_indices) == len(set(black_indices))

    def test_indices_in_valid_range(self) -> None:
        """Test that all indices are within valid range."""
        board = chess.Board()

        white_indices = board_to_halfkp_indices(board, is_white_pov=True)
        for idx in white_indices:
            assert 0 <= idx < HALFKP_FEATURES_PER_SIDE

        black_indices = board_to_halfkp_indices(board, is_white_pov=False)
        for idx in black_indices:
            assert 0 <= idx < HALFKP_FEATURES_PER_SIDE

    def test_empty_board_returns_empty_list(self) -> None:
        """Test that board with no pieces returns empty list."""
        # Set up a board with only kings
        board = chess.Board()
        board.set_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")

        white_indices = board_to_halfkp_indices(board, is_white_pov=True)
        black_indices = board_to_halfkp_indices(board, is_white_pov=False)

        # Only kings, no other pieces
        assert len(white_indices) == 0
        assert len(black_indices) == 0

    def test_after_move_count_changes(self) -> None:
        """Test that feature count changes after a move."""
        board = chess.Board()
        initial_count = len(board_to_halfkp_indices(board, is_white_pov=True))

        # Make a move (e2e4)
        board.push(chess.Move.from_uci("e2e4"))
        after_move_count = len(board_to_halfkp_indices(board, is_white_pov=True))

        # Count should be the same (no captures)
        assert initial_count == after_move_count

    def test_after_capture_count_decreases(self) -> None:
        """Test that feature count decreases after a capture."""
        board = chess.Board()
        # Set up a position with a capture available
        board.set_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        initial_count = len(board_to_halfkp_indices(board, is_white_pov=True))

        # Capture (e4xd5 is not legal, but d2d4 then exd4 would work)
        # Let's use a simpler position
        board.set_fen("rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2")
        board.push(chess.Move.from_uci("e5d4"))  # Black captures
        after_capture_count = len(board_to_halfkp_indices(board, is_white_pov=True))

        # Count should decrease by 1 (one piece captured)
        assert after_capture_count == initial_count - 1


class TestBoardToHalfKPTensor:
    """Test HalfKP tensor conversion."""

    def test_tensor_shape(self) -> None:
        """Test that tensors have correct shape."""
        board = chess.Board()
        white_tensor, black_tensor = board_to_halfkp_tensor(board)

        assert white_tensor.shape == (HALFKP_FEATURES_PER_SIDE,)
        assert black_tensor.shape == (HALFKP_FEATURES_PER_SIDE,)

    def test_tensor_dtype(self) -> None:
        """Test that tensors have correct dtype."""
        board = chess.Board()
        white_tensor, black_tensor = board_to_halfkp_tensor(board)

        assert white_tensor.dtype == torch.float32
        assert black_tensor.dtype == torch.float32

    def test_tensor_binary(self) -> None:
        """Test that tensors contain only 0s and 1s."""
        board = chess.Board()
        white_tensor, black_tensor = board_to_halfkp_tensor(board)

        assert torch.all((white_tensor == 0) | (white_tensor == 1))
        assert torch.all((black_tensor == 0) | (black_tensor == 1))

    def test_tensor_active_features_count(self) -> None:
        """Test that tensor has correct number of active features."""
        board = chess.Board()
        white_tensor, black_tensor = board_to_halfkp_tensor(board)

        # 30 pieces + 1 bias = 31 active features
        assert (white_tensor > 0).sum().item() == 31
        assert (black_tensor > 0).sum().item() == 31

    def test_bias_plane_always_active(self) -> None:
        """Test that bias plane (last feature) is always 1."""
        board = chess.Board()
        white_tensor, black_tensor = board_to_halfkp_tensor(board)

        assert white_tensor[-1] == 1.0
        assert black_tensor[-1] == 1.0


class TestBoardToInputTensor:
    """Test full input tensor conversion."""

    def test_tensor_shape(self) -> None:
        """Test that input tensor has correct shape."""
        board = chess.Board()
        tensor = board_to_input_tensor(board)

        assert tensor.shape == (TOTAL_FEATURES,)

    def test_tensor_dtype(self) -> None:
        """Test that input tensor has correct dtype."""
        board = chess.Board()
        tensor = board_to_input_tensor(board)

        assert tensor.dtype == torch.float32

    def test_tensor_binary(self) -> None:
        """Test that input tensor contains only 0s and 1s."""
        board = chess.Board()
        tensor = board_to_input_tensor(board)

        assert torch.all((tensor == 0) | (tensor == 1))

    def test_tensor_active_features_count(self) -> None:
        """Test that input tensor has correct number of active features."""
        board = chess.Board()
        tensor = board_to_input_tensor(board)

        # 30 pieces per side + 2 bias = 62 active features
        assert (tensor > 0).sum().item() == 62

    def test_concatenation(self) -> None:
        """Test that input tensor is concatenation of white and black tensors."""
        board = chess.Board()
        white_tensor, black_tensor = board_to_halfkp_tensor(board)
        full_tensor = board_to_input_tensor(board)

        # First half should match white tensor
        assert torch.allclose(full_tensor[:HALFKP_FEATURES_PER_SIDE], white_tensor)
        # Second half should match black tensor
        assert torch.allclose(full_tensor[HALFKP_FEATURES_PER_SIDE:], black_tensor)


class TestHalfKPConsistency:
    """Test consistency between C++ and Python implementations."""

    def test_cpp_and_python_match(self) -> None:
        """Test that C++ and Python implementations give same results."""
        board = chess.Board()

        # Get C++ results
        cpp_white = chess.halfkp.board_to_halfkp_indices(board, is_white_pov=True)
        cpp_black = chess.halfkp.board_to_halfkp_indices(board, is_white_pov=False)

        # Get Python results
        py_white = board_to_halfkp_indices(board, is_white_pov=True)
        py_black = board_to_halfkp_indices(board, is_white_pov=False)

        # Convert to sets for comparison (order doesn't matter)
        assert set(cpp_white) == set(py_white)
        assert set(cpp_black) == set(py_black)

    def test_tensor_conversion_matches(self) -> None:
        """Test that C++ tensor conversion matches Python."""
        board = chess.Board()

        # Get C++ tensor
        cpp_tensor = torch.tensor(
            chess.halfkp.board_to_input_tensor(board), dtype=torch.float32
        )

        # Get Python tensor
        py_tensor = board_to_input_tensor(board)

        # Should be identical
        assert torch.allclose(cpp_tensor, py_tensor)


class TestHalfKPEdgeCases:
    """Test edge cases and special positions."""

    def test_position_with_promotions(self) -> None:
        """Test position with promoted pieces."""
        board = chess.Board()
        # Set up a position with a promoted queen
        board.set_fen("4k3/8/8/8/8/8/7P/4K2Q w - - 0 1")

        white_indices = board_to_halfkp_indices(board, is_white_pov=True)
        # 1 pawn + 1 queen = 2 pieces
        assert len(white_indices) == 2

    def test_position_with_many_pieces(self) -> None:
        """Test position with many pieces."""
        board = chess.Board()
        # Starting position has maximum non-king pieces
        white_indices = board_to_halfkp_indices(board, is_white_pov=True)
        assert len(white_indices) == 30

    def test_endgame_position(self) -> None:
        """Test sparse endgame position."""
        board = chess.Board()
        board.set_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")

        white_indices = board_to_halfkp_indices(board, is_white_pov=True)
        # Only kings, no other pieces
        assert len(white_indices) == 0

    def test_different_king_positions(self) -> None:
        """Test that king position affects feature indices."""
        # Position with king on e1
        board1 = chess.Board()
        board1.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        indices1 = board_to_halfkp_indices(board1, is_white_pov=True)

        # Position with king on e2 (same pieces, different king position)
        board2 = chess.Board()
        board2.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPKPPP/RNBQ1BNR w kq - 0 1")
        indices2 = board_to_halfkp_indices(board2, is_white_pov=True)

        # Indices should be different (king position affects all features)
        assert set(indices1) != set(indices2)
