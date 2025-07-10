from typing import TYPE_CHECKING

import chess

from src.engine.constants import PIECE_VALUES

if TYPE_CHECKING:
    from src.engine.search.transposition_table import TranspositionTable
    from src.engine.search.zobrist import Zobrist


class MoveOrderer:
    """
    Handles the ordering of moves to improve alpha-beta pruning efficiency.
    """

    MVV_LVA_MULTIPLIER = 10

    def __init__(
        self,
        board: chess.Board,
        zobrist: "Zobrist | None" = None,
        transposition_table: "TranspositionTable | None" = None,
    ):
        """
        Initialize the move orderer.

        Args:
            board: Chess board position
            zobrist: Optional Zobrist hasher
            transposition_table: Optional transposition table
        """
        self.board = board
        self.zobrist = zobrist
        self.transposition_table = transposition_table

    def order_moves(self, moves: list[chess.Move]) -> list[chess.Move]:
        """
        Order moves to improve alpha-beta pruning efficiency.
        Prioritizes:
        1. PV move from transposition table
        2. Captures ordered by MVV-LVA
        3. Promotions
        4. Other moves

        Args:
            moves: List of legal moves to order

        Returns:
            Ordered list of moves
        """
        # Get the current position hash
        position_hash: int | None = None
        if self.zobrist:
            position_hash = self.zobrist.get_current_hash()

        # Initialize the PV move from the transposition table
        pv_move: chess.Move | None = None
        if self.transposition_table and position_hash is not None:
            pv_move = self.transposition_table.get_best_move(position_hash)

        # Assign scores to moves for ordering
        move_scores: list[tuple[chess.Move, int]] = []

        for move in moves:
            score = 0

            # PV move gets highest score
            if pv_move and move == pv_move:
                score = 10000

            # Captures get scored by MVV-LVA
            elif self.board.is_capture(move):
                victim_value = self._get_piece_value(move.to_square)
                aggressor_value = self._get_piece_value(move.from_square)
                if victim_value and aggressor_value:
                    # MVV-LVA: Most Valuable Victim - Least Valuable Aggressor
                    score = self.MVV_LVA_MULTIPLIER * victim_value - aggressor_value

            # Promotions
            elif move.promotion:
                promotion_value = int(PIECE_VALUES[move.promotion])
                pawn_value = int(PIECE_VALUES[chess.PAWN])
                score = promotion_value - pawn_value

            move_scores.append((move, score))

        # Sort by score in descending order
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]

    def _get_piece_value(self, square: int) -> int:
        """Get the value of a piece at a given square."""
        piece = self.board.piece_at(square)
        if piece:
            return int(PIECE_VALUES.get(piece.piece_type, 0))
        return 0
