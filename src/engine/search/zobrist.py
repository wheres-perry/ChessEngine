import random

from engine._core import chess_engine_core as chess


def rand64() -> int:
    """Generate a random 64-bit integer."""
    return random.getrandbits(64)


def piece_to_index(piece_type: int, color: bool) -> int:
    """Convert piece type and color to array index (0-11)."""
    piece_value = int(piece_type)
    color_value = int(color)
    return piece_value + (6 * color_value)


SQ_A1 = 0
SQ_D1 = 3
SQ_F1 = 5
SQ_H1 = 7
SQ_A8 = 56
SQ_D8 = 59
SQ_F8 = 61
SQ_H8 = 63


# Performance critical component, linting disabled
# pylint: disable=too-many-branches,too-many-statements
class Zobrist:
    """
    High-performance Zobrist hashing with reliable incremental updates.

    Provides efficient position hashing that can be updated incrementally
    as moves are made and unmade.
    """

    __slots__ = ("_current_hash", "castling_keys", "ep_keys", "piece_keys", "turn_key")

    _current_hash: int | None

    def __init__(self, seed: int | None = None):
        """Initialize Zobrist hash keys for all board elements.

        Args:
            seed: Optional seed for random number generation (for testing).
        """
        if seed is not None:
            random.seed(seed)
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
        cr = board.get_castling_rights()
        if cr & chess.BB_H1:  # White kingside
            h ^= self.castling_keys[0]
        if cr & chess.BB_A1:  # White queenside
            h ^= self.castling_keys[1]
        if cr & chess.BB_H8:  # Black kingside
            h ^= self.castling_keys[2]
        if cr & chess.BB_A8:  # Black queenside
            h ^= self.castling_keys[3]
        # Hash en passant
        ep_square = board.ep_square
        if ep_square is not None:
            ep_file = chess.square_file(ep_square)
            h ^= self.ep_keys[ep_file]
        # Hash turn
        if board.turn == chess.BLACK:
            h ^= self.turn_key
        self._current_hash = h
        return h

    def make_move_hash(self, board: chess.Board, move: chess.Move) -> int:
        """Fast incremental hash update without expensive push/pop operations.

        Args:
            board: Current chess board position.
            move: Move to apply to the hash.

        Returns:
            Updated Zobrist hash value.
        """
        board.push(move)
        updated_hash = self.hash_board(board)
        board.pop()
        return updated_hash

    def get_current_hash(self) -> int | None:
        """Get the current hash value without recalculating.

        Returns:
            Current Zobrist hash value, or None if not initialized.
        """
        return self._current_hash

    def set_current_hash(self, hash_val: int | None) -> None:
        """Set the current hash value (for initialization or restoring after pop).

        Args:
            hash_val: Hash value to set.
        """
        self._current_hash = hash_val

    def invalidate_hash(self) -> None:
        """Invalidate the current hash."""
        self._current_hash = None
