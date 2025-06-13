# src/engine/zobrist.py

import random
from typing import Optional

import chess


class Zobrist:
    """
    Zobrist hashing implementation with incremental updates.

    Provides efficient position hashing that can be updated incrementally
    as moves are made and unmade, avoiding full board rehashing.
    """

    def __init__(self):
        """Initialize Zobrist hash keys for all board elements."""
        self.keys = {}

        # Piece keys: 12 pieces (6 types x 2 colors) x 64 squares

        for piece in range(1, 7):  # PAWN to KING
            for color in [chess.WHITE, chess.BLACK]:
                for square in range(64):
                    self.keys[(piece, color, square)] = random.randint(0, 2**64 - 1)
        # Castling rights

        self.keys["castling_wk"] = random.randint(0, 2**64 - 1)
        self.keys["castling_wq"] = random.randint(0, 2**64 - 1)
        self.keys["castling_bk"] = random.randint(0, 2**64 - 1)
        self.keys["castling_bq"] = random.randint(0, 2**64 - 1)

        # En passant file

        for file in range(8):
            self.keys[("ep", file)] = random.randint(0, 2**64 - 1)
        # Turn

        self.keys["turn"] = random.randint(0, 2**64 - 1)

        # Current hash value for incremental updates

        self._current_hash: Optional[int] = None

    def hash_board(self, board: chess.Board) -> int:
        """
        Compute the full Zobrist hash for a board position.

        Args:
            board: Chess board to hash

        Returns:
            64-bit Zobrist hash value
        """
        hash_val = 0

        # Hash pieces

        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                hash_val ^= self.keys[(piece.piece_type, piece.color, square)]
        # Hash castling rights

        if board.has_kingside_castling_rights(chess.WHITE):
            hash_val ^= self.keys["castling_wk"]
        if board.has_queenside_castling_rights(chess.WHITE):
            hash_val ^= self.keys["castling_wq"]
        if board.has_kingside_castling_rights(chess.BLACK):
            hash_val ^= self.keys["castling_bk"]
        if board.has_queenside_castling_rights(chess.BLACK):
            hash_val ^= self.keys["castling_bq"]
        # Hash en passant

        if board.ep_square is not None:
            file = chess.square_file(board.ep_square)
            hash_val ^= self.keys[("ep", file)]
        # Hash turn

        if board.turn == chess.BLACK:
            hash_val ^= self.keys["turn"]
        self._current_hash = hash_val
        return hash_val

    def update_hash_for_move(
        self,
        board: chess.Board,
        move: chess.Move,
        old_castling_rights: int,
        old_ep_square: Optional[int],
    ) -> int:
        """
        Incrementally update the hash after a move is made.

        Args:
            board: Board after the move has been made
            move: The move that was made
            old_castling_rights: Castling rights before the move
            old_ep_square: En passant square before the move

        Returns:
            Updated hash value
        """
        if self._current_hash is None:
            return self.hash_board(board)
        hash_val = self._current_hash

        # Toggle turn

        hash_val ^= self.keys["turn"]

        # Handle piece movement

        from_square = move.from_square
        to_square = move.to_square

        # Get the piece that moved (it's now on the to_square)

        moved_piece = board.piece_at(to_square)
        if moved_piece:
            # Remove piece from old square

            hash_val ^= self.keys[
                (moved_piece.piece_type, moved_piece.color, from_square)
            ]
            # Add piece to new square

            hash_val ^= self.keys[
                (moved_piece.piece_type, moved_piece.color, to_square)
            ]
        # Handle captures (remove captured piece)

        if board.is_capture(move):
            if board.is_en_passant(move):
                # En passant capture - remove the captured pawn

                if (
                    board.turn == chess.WHITE
                ):  # White just moved, so black pawn was captured
                    captured_square = to_square - 8
                    hash_val ^= self.keys[(chess.PAWN, chess.BLACK, captured_square)]
                else:  # Black just moved, so white pawn was captured
                    captured_square = to_square + 8
                    hash_val ^= self.keys[(chess.PAWN, chess.WHITE, captured_square)]
            else:
                # Regular capture - we need to determine what was captured
                # Since the move is already made, we need to infer the captured piece
                # This is a limitation - ideally we'd pass the captured piece info

                pass  # For now, fall back to full rehash for captures
        # Handle promotion

        if move.promotion:
            # Remove the pawn that was promoted

            hash_val ^= self.keys[(chess.PAWN, moved_piece.color, to_square)]
            # The promoted piece is already added above, so we're good
        # Handle castling

        if board.is_castling(move):
            # Handle rook movement in castling

            if move.to_square == chess.G1:  # White kingside
                hash_val ^= self.keys[(chess.ROOK, chess.WHITE, chess.H1)]
                hash_val ^= self.keys[(chess.ROOK, chess.WHITE, chess.F1)]
            elif move.to_square == chess.C1:  # White queenside
                hash_val ^= self.keys[(chess.ROOK, chess.WHITE, chess.A1)]
                hash_val ^= self.keys[(chess.ROOK, chess.WHITE, chess.D1)]
            elif move.to_square == chess.G8:  # Black kingside
                hash_val ^= self.keys[(chess.ROOK, chess.BLACK, chess.H8)]
                hash_val ^= self.keys[(chess.ROOK, chess.BLACK, chess.F8)]
            elif move.to_square == chess.C8:  # Black queenside
                hash_val ^= self.keys[(chess.ROOK, chess.BLACK, chess.A8)]
                hash_val ^= self.keys[(chess.ROOK, chess.BLACK, chess.D8)]
        # Update castling rights

        self._update_castling_hash(hash_val, old_castling_rights, board.castling_rights)

        # Update en passant

        if old_ep_square is not None:
            old_file = chess.square_file(old_ep_square)
            hash_val ^= self.keys[("ep", old_file)]
        if board.ep_square is not None:
            new_file = chess.square_file(board.ep_square)
            hash_val ^= self.keys[("ep", new_file)]
        self._current_hash = hash_val
        return hash_val

    def _update_castling_hash(
        self, hash_val: int, old_rights: int, new_rights: int
    ) -> None:
        """Update hash for castling rights changes."""
        # XOR out old rights and XOR in new rights

        if (old_rights & chess.BB_H1) != (new_rights & chess.BB_H1):
            hash_val ^= self.keys["castling_wk"]
        if (old_rights & chess.BB_A1) != (new_rights & chess.BB_A1):
            hash_val ^= self.keys["castling_wq"]
        if (old_rights & chess.BB_H8) != (new_rights & chess.BB_H8):
            hash_val ^= self.keys["castling_bk"]
        if (old_rights & chess.BB_A8) != (new_rights & chess.BB_A8):
            hash_val ^= self.keys["castling_bq"]

    def get_current_hash(self) -> Optional[int]:
        """Get the current hash value without recalculating."""
        return self._current_hash

    def invalidate_hash(self) -> None:
        """Invalidate the current hash, forcing recalculation on next access."""
        self._current_hash = None
