import random

import chess

rand64 = lambda: random.getrandbits(64)


class Zobrist:
    """
    Zobrist hashing implementation with incremental updates.

    Provides efficient position hashing that can be updated incrementally
    as moves are made and unmade, avoiding full board rehashing.
    """

    __slots__ = ("piece_keys", "castling_keys", "ep_keys", "turn_key", "_current_hash")

    def __init__(self):
        """Initialize Zobrist hash keys for all board elements."""
        # 12×64 piece table (6 piece types × 2 colors × 64 squares)

        self.piece_keys = [[rand64() for _ in range(64)] for _ in range(12)]

        # [W-K, W-Q, B-K, B-Q] castling rights

        self.castling_keys = [rand64() for _ in range(4)]

        # Files a-h for en passant

        self.ep_keys = [rand64() for _ in range(8)]

        # Side to move

        self.turn_key = rand64()

        # Current hash value for incremental updates

        self._current_hash = None

    def hash_board(self, board: chess.Board) -> int:
        """
        Compute the full Zobrist hash for a board position.

        Args:
            board: Chess board to hash

        Returns:
            64-bit Zobrist hash value
        """
        h = 0

        # Hash pieces - iterate only over actual pieces

        for piece_type in chess.PIECE_TYPES:
            # White pieces

            for square in board.pieces(piece_type, chess.WHITE):
                idx = (piece_type - 1) * 2
                h ^= self.piece_keys[idx][square]
            # Black pieces

            for square in board.pieces(piece_type, chess.BLACK):
                idx = (piece_type - 1) * 2 + 1
                h ^= self.piece_keys[idx][square]
        # Hash castling rights

        cr = board.castling_rights
        if cr & chess.BB_H1:
            h ^= self.castling_keys[0]  # White kingside
        if cr & chess.BB_A1:
            h ^= self.castling_keys[1]  # White queenside
        if cr & chess.BB_H8:
            h ^= self.castling_keys[2]  # Black kingside
        if cr & chess.BB_A8:
            h ^= self.castling_keys[3]  # Black queenside
        # Hash en passant

        if board.ep_square is not None:
            h ^= self.ep_keys[chess.square_file(board.ep_square)]
        # Hash turn

        if board.turn == chess.BLACK:
            h ^= self.turn_key
        self._current_hash = h
        return h

    def update_hash_for_move(
        self,
        post_move_board: chess.Board,
        move: chess.Move,
        old_castling_rights: int,
        old_ep_square: int | None,
        captured_piece_type: int | None,
        was_ep: bool,
        ks_castle: bool,
        qs_castle: bool,
    ) -> int:
        """
        Incrementally update the hash after a move is made.

        Args:
            post_move_board: Board after the move has been made
            move: The move that was made
            old_castling_rights: Castling rights before the move
            old_ep_square: En passant square before the move
            captured_piece_type: Type of piece captured (if any)
            was_ep: True if the move was an en passant capture
            ks_castle: True if the move was kingside castling
            qs_castle: True if the move was queenside castling

        Returns:
            Updated hash value
        """
        # Pull into locals for speed

        h = self._current_hash
        if h is None:
            # This shouldn't happen in normal operation

            return self.hash_board(post_move_board)
        pk = self.piece_keys
        ck = self.castling_keys
        epk = self.ep_keys
        tkey = self.turn_key

        # 1) Flip turn

        h ^= tkey

        # 2) Move piece (incl. promotion)

        mover = not post_move_board.turn  # Color of piece that moved
        if move.promotion:
            # Pawn out, promoted piece in

            h ^= pk[0 * 2 + mover][move.from_square]  # Remove pawn from source square
            promo_idx = (move.promotion - 1) * 2 + mover
            h ^= pk[promo_idx][move.to_square]  # Add promoted piece at destination
        else:
            # Find moved piece from post-move board

            piece = post_move_board.piece_at(move.to_square)
            if piece:  # This should always be true for a regular move
                idx = (piece.piece_type - 1) * 2 + mover
                # XOR out from old square, XOR in to new square

                h ^= pk[idx][move.from_square] ^ pk[idx][move.to_square]
        # 3) Handle capture

        if captured_piece_type:
            cap_color = post_move_board.turn  # Color of captured piece
            cap_idx = (captured_piece_type - 1) * 2 + cap_color
            if was_ep:
                # Pawn captured behind the to-square

                cap_sq = move.to_square + (8 if mover == chess.WHITE else -8)
                h ^= pk[0 * 2 + cap_color][
                    cap_sq
                ]  # Remove captured pawn (always a pawn for EP)
            else:
                # Regular capture - remove captured piece from destination square

                h ^= pk[cap_idx][move.to_square]
        # 4) Handle rook movement in castling

        if ks_castle:
            # Get correct rook squares based on color

            rook_src = chess.H1 if mover == chess.WHITE else chess.H8
            rook_dst = chess.F1 if mover == chess.WHITE else chess.F8
            r_idx = (chess.ROOK - 1) * 2 + mover
            h ^= pk[r_idx][rook_src] ^ pk[r_idx][rook_dst]
        elif qs_castle:
            # Get correct rook squares based on color

            rook_src = chess.A1 if mover == chess.WHITE else chess.A8
            rook_dst = chess.D1 if mover == chess.WHITE else chess.D8
            r_idx = (chess.ROOK - 1) * 2 + mover
            h ^= pk[r_idx][rook_src] ^ pk[r_idx][rook_dst]
        # 5) Update castling rights

        new_cr = post_move_board.castling_rights
        if (old_castling_rights & chess.BB_H1) != (new_cr & chess.BB_H1):
            h ^= ck[0]
        if (old_castling_rights & chess.BB_A1) != (new_cr & chess.BB_A1):
            h ^= ck[1]
        if (old_castling_rights & chess.BB_H8) != (new_cr & chess.BB_H8):
            h ^= ck[2]
        if (old_castling_rights & chess.BB_A8) != (new_cr & chess.BB_A8):
            h ^= ck[3]
        # 6) Update en passant square

        if old_ep_square is not None:
            h ^= epk[chess.square_file(old_ep_square)]
        if post_move_board.ep_square is not None:
            h ^= epk[chess.square_file(post_move_board.ep_square)]
        self._current_hash = h
        return h

    def get_current_hash(self) -> None | int:
        """Get the current hash value without recalculating."""
        return self._current_hash

    def set_current_hash(self, hash_val: int | None) -> None:
        """Set the current hash value. Used for initialization or restoring after pop."""
        self._current_hash = hash_val

    def invalidate_hash(self) -> None:
        """Invalidate the current hash, typically not needed if set_current_hash is used on pop."""
        self._current_hash = None
