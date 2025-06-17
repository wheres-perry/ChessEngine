import random

import chess

rand64 = lambda: random.getrandbits(64)


def piece_to_index(piece_type: int, color: bool) -> int:
    """Convert piece type and color to array index (0-11)."""
    return (piece_type - 1) + (6 * (1 if color else 0))


class Zobrist:
    """
    Zobrist hashing implementation with incremental updates.

    Provides efficient position hashing that can be updated incrementally
    as moves are made and unmade, avoiding full board rehashing.
    """

    __slots__ = ("piece_keys", "castling_keys", "ep_keys", "turn_key", "_current_hash")

    def __init__(self, seed=None):
        """Initialize Zobrist hash keys for all board elements."""
        if seed is not None:
            random.seed(seed)
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
            for color in [chess.WHITE, chess.BLACK]:
                for square in board.pieces(piece_type, color):
                    piece_index = piece_to_index(piece_type, color)
                    h ^= self.piece_keys[piece_index][square]
        # Hash castling rights

        cr = board.castling_rights
        if cr & chess.BB_H1:  # White kingside
            h ^= self.castling_keys[0]
        if cr & chess.BB_A1:  # White queenside
            h ^= self.castling_keys[1]
        if cr & chess.BB_H8:  # Black kingside
            h ^= self.castling_keys[2]
        if cr & chess.BB_A8:  # Black queenside
            h ^= self.castling_keys[3]
        # Hash en passant

        if board.ep_square is not None:
            ep_file = chess.square_file(board.ep_square)
            h ^= self.ep_keys[ep_file]
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
        """
        # Pull into locals for speed

        h = self._current_hash
        if h is None:
            return self.hash_board(post_move_board)
        pk = self.piece_keys
        ck = self.castling_keys
        epk = self.ep_keys
        tkey = self.turn_key

        # 1) Flip turn

        h ^= tkey

        # Determine mover color and captured piece color

        mover = not post_move_board.turn  # Color of piece that moved
        cap_color = not mover  # Color of captured piece

        # 2) Move piece (incl. promotion)

        if move.promotion:
            # Remove original pawn

            pawn_index = piece_to_index(chess.PAWN, mover)
            h ^= pk[pawn_index][move.from_square]

            # Add promoted piece

            promoted_index = piece_to_index(move.promotion, mover)
            h ^= pk[promoted_index][move.to_square]
        else:
            # Normal move - get piece type that moved

            moved_piece = post_move_board.piece_at(move.to_square)
            if moved_piece:
                piece_index = piece_to_index(moved_piece.piece_type, mover)
                h ^= pk[piece_index][move.from_square]  # Remove from old square
                h ^= pk[piece_index][move.to_square]  # Add to new square
        # 3) Handle capture

        if captured_piece_type and not was_ep:
            # Normal capture - remove captured piece

            cap_index = piece_to_index(captured_piece_type, cap_color)
            h ^= pk[cap_index][move.to_square]
        elif was_ep:
            # En passant capture - remove pawn from different square
            # FIX: The captured pawn is on the same rank as the moving pawn

            if mover == chess.WHITE:
                ep_capture_square = move.to_square - 8  # Black pawn one rank below
            else:
                ep_capture_square = move.to_square + 8  # White pawn one rank above
            cap_index = piece_to_index(chess.PAWN, cap_color)
            h ^= pk[cap_index][ep_capture_square]
        # 4) Handle rook movement in castling

        if ks_castle:
            rook_index = piece_to_index(chess.ROOK, mover)
            if mover == chess.WHITE:
                h ^= pk[rook_index][chess.H1]  # Remove from h1
                h ^= pk[rook_index][chess.F1]  # Add to f1
            else:
                h ^= pk[rook_index][chess.H8]  # Remove from h8
                h ^= pk[rook_index][chess.F8]  # Add to f8
        elif qs_castle:
            rook_index = piece_to_index(chess.ROOK, mover)
            if mover == chess.WHITE:
                h ^= pk[rook_index][chess.A1]  # Remove from a1
                h ^= pk[rook_index][chess.D1]  # Add to d1
            else:
                h ^= pk[rook_index][chess.A8]  # Remove from a8
                h ^= pk[rook_index][chess.D8]  # Add to d8
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
            old_ep_file = chess.square_file(old_ep_square)
            h ^= epk[old_ep_file]
        if post_move_board.ep_square is not None:
            new_ep_file = chess.square_file(post_move_board.ep_square)
            h ^= epk[new_ep_file]
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
