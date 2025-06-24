import random
import chess

rand64 = lambda: random.getrandbits(64)

def piece_to_index(piece_type: int, color: bool) -> int:
    """Convert piece type and color to array index (0-11)."""
    return (piece_type - 1) + (6 * (1 if color else 0))

class Zobrist:
    """
    High-performance Zobrist hashing with reliable incremental updates.
    Provides efficient position hashing that can be updated incrementally
    as moves are made and unmade.
    """

    __slots__ = ("piece_keys", "castling_keys", "ep_keys", "turn_key", "_current_hash")

    def __init__(self, seed=None):
        """Initialize Zobrist hash keys for all board elements."""
        if seed is not None:
            random.seed(seed)

        # 12Ã—64 piece table (6 piece types Ã— 2 colors Ã— 64 squares)
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

    def make_move_hash(self, board: chess.Board, move: chess.Move) -> int:
        """FIXED: Fast incremental hash without expensive push/pop operations."""
        if self._current_hash is None:
            return self.hash_board(board)

        h = self._current_hash

        # 1. Flip turn
        h ^= self.turn_key

        # 2. Get piece information
        moving_piece = board.piece_at(move.from_square)
        if not moving_piece:
            # Fallback only if really needed
            board.push(move)
            result = self.hash_board(board)
            board.pop()
            return result

        captured_piece = board.piece_at(move.to_square)

        # 3. Handle the moving piece
        if move.promotion:
            pawn_index = piece_to_index(chess.PAWN, moving_piece.color)
            h ^= self.piece_keys[pawn_index][move.from_square]
            promoted_index = piece_to_index(move.promotion, moving_piece.color)
            h ^= self.piece_keys[promoted_index][move.to_square]
        else:
            piece_index = piece_to_index(moving_piece.piece_type, moving_piece.color)
            h ^= self.piece_keys[piece_index][move.from_square]
            h ^= self.piece_keys[piece_index][move.to_square]

        # 4. Handle captures
        if captured_piece and not board.is_en_passant(move):
            cap_index = piece_to_index(captured_piece.piece_type, captured_piece.color)
            h ^= self.piece_keys[cap_index][move.to_square]
        elif board.is_en_passant(move):
            ep_capture_square = move.to_square + (-8 if moving_piece.color else 8)
            cap_index = piece_to_index(chess.PAWN, not moving_piece.color)
            h ^= self.piece_keys[cap_index][ep_capture_square]

        # 5. Handle castling rook movement
        if board.is_castling(move):
            rook_index = piece_to_index(chess.ROOK, moving_piece.color)
            if board.is_kingside_castling(move):
                if moving_piece.color:  # White
                    h ^= self.piece_keys[rook_index][chess.H1]
                    h ^= self.piece_keys[rook_index][chess.F1]
                else:  # Black
                    h ^= self.piece_keys[rook_index][chess.H8]
                    h ^= self.piece_keys[rook_index][chess.F8]
            else:  # Queenside
                if moving_piece.color:  # White
                    h ^= self.piece_keys[rook_index][chess.A1]
                    h ^= self.piece_keys[rook_index][chess.D1]
                else:  # Black
                    h ^= self.piece_keys[rook_index][chess.A8]
                    h ^= self.piece_keys[rook_index][chess.D8]

        # FIXED: 6. Handle castling rights efficiently WITHOUT push/pop
        old_cr = board.castling_rights
        new_cr = old_cr
        
        # Calculate new castling rights based on move
        if moving_piece.piece_type == chess.KING:
            if moving_piece.color:  # White king
                new_cr &= ~(chess.BB_A1 | chess.BB_H1)
            else:  # Black king
                new_cr &= ~(chess.BB_A8 | chess.BB_H8)
        elif moving_piece.piece_type == chess.ROOK:
            if move.from_square == chess.A1:
                new_cr &= ~chess.BB_A1
            elif move.from_square == chess.H1:
                new_cr &= ~chess.BB_H1
            elif move.from_square == chess.A8:
                new_cr &= ~chess.BB_A8
            elif move.from_square == chess.H8:
                new_cr &= ~chess.BB_H8
    
        # Rook captures
        if move.to_square == chess.A1:
            new_cr &= ~chess.BB_A1
        elif move.to_square == chess.H1:
            new_cr &= ~chess.BB_H1
        elif move.to_square == chess.A8:
            new_cr &= ~chess.BB_A8
        elif move.to_square == chess.H8:
            new_cr &= ~chess.BB_H8

        # Update hash for castling changes
        if (old_cr & chess.BB_H1) != (new_cr & chess.BB_H1):
            h ^= self.castling_keys[0]
        if (old_cr & chess.BB_A1) != (new_cr & chess.BB_A1):
            h ^= self.castling_keys[1]
        if (old_cr & chess.BB_H8) != (new_cr & chess.BB_H8):
            h ^= self.castling_keys[2]
        if (old_cr & chess.BB_A8) != (new_cr & chess.BB_A8):
            h ^= self.castling_keys[3]

        # FIXED: 7. Handle en passant efficiently WITHOUT push/pop
        old_ep = board.ep_square
        new_ep = None
        
        # Calculate new en passant square
        if (moving_piece.piece_type == chess.PAWN and 
            abs(move.to_square - move.from_square) == 16):
            new_ep = move.from_square + (8 if moving_piece.color else -8)

        # Update hash for en passant changes
        if old_ep is not None:
            h ^= self.ep_keys[chess.square_file(old_ep)]
        if new_ep is not None:
            h ^= self.ep_keys[chess.square_file(new_ep)]

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
        Update hash after a move has been made on the board.
        This method is kept for compatibility but simplified.
        """
        self._current_hash = self.hash_board(post_move_board)
        return self._current_hash

    def get_current_hash(self) -> None | int:
        """Get the current hash value without recalculating."""
        return self._current_hash

    def set_current_hash(self, hash_val: int | None) -> None:
        """Set the current hash value. Used for initialization or restoring after pop."""
        self._current_hash = hash_val

    def invalidate_hash(self) -> None:
        """Invalidate the current hash, typically not needed if set_current_hash is used on pop."""
        self._current_hash = None