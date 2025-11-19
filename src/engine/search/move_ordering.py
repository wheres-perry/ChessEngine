"""
Modular move ordering for alpha-beta search.

Orders moves to maximize alpha-beta cutoffs by respecting configuration flags.
Supports multiple heuristics from Tree 1 (Move Exploration):
- Hash move ordering (requires TT)
- MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
- SEE ordering (Static Exchange Evaluation)
- Killer moves
- History heuristic
- Countermove heuristic
"""

from typing import TYPE_CHECKING

from engine._core import chess_engine_core as chess
from src.engine.config import SearchConfig
from src.engine.constants import PIECE_VALUES

if TYPE_CHECKING:
    from src.engine.search.transposition_table import TranspositionTable
    from src.engine.search.zobrist import Zobrist


class MoveOrderer:
    """
    Modular move orderer that respects Tree 1 dependencies.

    Ordering heuristics are applied based on configuration:
    - Hash move (requires use_hash_move_ordering + TT)
    - MVV-LVA (requires use_mvv_lva)
    - SEE (requires use_see_ordering)
    - Killer moves (requires use_killer_moves + alpha-beta)
    - History heuristic (requires use_history_heuristic + alpha-beta)
    - Countermove (requires use_countermove_heuristic + alpha-beta)
    """

    MVV_LVA_MULTIPLIER = 10
    KILLER_SCORE = 9000
    HISTORY_MAX_SCORE = 5000

    def __init__(
        self,
        board: chess.Board,
        config: SearchConfig,
        zobrist: "Zobrist | None" = None,
        transposition_table: "TranspositionTable | None" = None,
    ):
        """
        Initialize the modular move orderer.

        Args:
            board: Chess board position
            config: Search configuration specifying which heuristics to use
            zobrist: Optional Zobrist hasher
            transposition_table: Optional transposition table
        """
        self.board = board
        self.config = config
        self.zobrist = zobrist
        self.transposition_table = transposition_table

        # Initialize killer moves table (2 per depth)
        self.killer_moves: dict[int, list[chess.Move]] = {}
        self.max_killers_per_depth = 2

        # Initialize history heuristic table
        # [from_square][to_square] -> success count
        self.history_table: list[list[int]] = [[0] * 64 for _ in range(64)]

        # Initialize countermove table
        # [from_square][to_square] -> best response move
        self.countermove_table: dict[tuple[int, int], chess.Move] = {}

    def order_moves(
        self,
        moves: list[chess.Move],
        depth: int = 0,
        last_move: chess.Move | None = None,
    ) -> list[chess.Move]:
        """
        Order moves to improve alpha-beta pruning efficiency.

        Applies heuristics based on configuration flags.

        Args:
            moves: List of legal moves to order
            depth: Current search depth (for killer moves)
            last_move: Previous opponent move (for countermove heuristic)

        Returns:
            Ordered list of moves
        """
        if not moves:
            return []

        # Get PV move from TT (if hash move ordering enabled)
        pv_move = self._get_pv_move()

        # Assign scores to moves
        move_scores: list[tuple[chess.Move, float]] = []

        for move in moves:
            score = self._score_move(move, depth, last_move, pv_move)
            move_scores.append((move, score))

        # Sort by score in descending order
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]

    def _get_pv_move(self) -> chess.Move | None:
        """Retrieve the PV move from the transposition table if available."""
        if not (
            self.config.use_hash_move_ordering
            and self.zobrist
            and self.transposition_table
        ):
            return None

        position_hash = self.zobrist.get_current_hash()
        if position_hash is None:
            return None

        return self.transposition_table.get_best_move(position_hash)

    def _score_move(  # noqa: PLR0911
        self,
        move: chess.Move,
        depth: int,
        last_move: chess.Move | None,
        pv_move: chess.Move | None,
    ) -> float:
        """Calculate the ordering score for a single move."""
        # 1. PV move from TT (highest priority)
        if pv_move and move == pv_move:
            return 10000.0

        # 2. Captures
        if self.board.is_capture(move):
            if self.config.use_mvv_lva:
                return self._score_mvv_lva(move)
            # Basic capture scoring
            return 8000.0 + self._get_piece_value(move.to_square)

        # 3. Killer moves (non-captures that caused beta cutoffs)
        if self.config.use_killer_moves and self._is_killer(move, depth):
            return self.KILLER_SCORE

        # 4. Countermove (response to opponent's last move)
        if (
            self.config.use_countermove_heuristic
            and last_move
            and self._is_countermove(move, last_move)
        ):
            return 7000.0

        # 5. History heuristic (historical success of this move)
        if self.config.use_history_heuristic:
            return self._get_history_score(move)

        # 6. Promotions
        if move.promotion:
            promotion_value = int(PIECE_VALUES[move.promotion])
            pawn_value = int(PIECE_VALUES[chess.PAWN])
            return promotion_value - pawn_value + 6000.0

        # 7. Default quiet move score
        return 0.0

    # =========================================================================
    # Scoring Methods
    # =========================================================================

    def _score_mvv_lva(self, move: chess.Move) -> float:
        """Score capture using MVV-LVA heuristic."""
        victim_value = self._get_piece_value(move.to_square)
        aggressor_value = self._get_piece_value(move.from_square)

        if victim_value and aggressor_value:
            # Higher score for capturing valuable pieces with cheap pieces
            return 8000.0 + self.MVV_LVA_MULTIPLIER * victim_value - aggressor_value

        return 8000.0

    def _get_history_score(self, move: chess.Move) -> float:
        """Get history heuristic score for move."""
        history_count = self.history_table[move.from_square][move.to_square]
        # Normalize to reasonable range
        return float(min(history_count, self.HISTORY_MAX_SCORE))

    # =========================================================================
    # Killer Moves (Node E)
    # =========================================================================

    def _is_killer(self, move: chess.Move, depth: int) -> bool:
        """Check if move is a killer move at this depth."""
        if depth not in self.killer_moves:
            return False
        return move in self.killer_moves[depth]

    def add_killer_move(self, move: chess.Move, depth: int) -> None:
        """Add a killer move at the given depth.

        Args:
            move: The move that caused a cutoff.
            depth: The search depth where this move caused a cutoff.
        """
        if not self.config.use_killer_moves:
            return

        # Don't store captures as killers (they're already well-ordered)
        if self.board.is_capture(move):
            return

        if depth not in self.killer_moves:
            self.killer_moves[depth] = []

        # Add move if not already present
        if move not in self.killer_moves[depth]:
            self.killer_moves[depth].insert(0, move)

            # Keep only the N most recent killers
            if len(self.killer_moves[depth]) > self.max_killers_per_depth:
                self.killer_moves[depth].pop()

    # =========================================================================
    # History Heuristic (Node F)
    # =========================================================================

    def update_history(self, move: chess.Move, depth: int) -> None:
        """Update history heuristic for a move that caused a cutoff.

        Args:
            move: The move that caused a cutoff.
            depth: The search depth (used for bonus calculation).
        """
        if not self.config.use_history_heuristic:
            return

        # Increment history score (depth bonus for deeper cutoffs)
        bonus = depth * depth  # Quadratic bonus
        self.history_table[move.from_square][move.to_square] += bonus

    def age_history(self) -> None:
        """Age the history table to give more weight to recent results."""
        if not self.config.use_history_heuristic:
            return

        # Divide all scores by 2 (simple aging)
        for i in range(64):
            for j in range(64):
                self.history_table[i][j] //= 2

    # =========================================================================
    # Countermove Heuristic (Node F)
    # =========================================================================

    def _is_countermove(self, move: chess.Move, last_move: chess.Move) -> bool:
        """Check if move is the best countermove to last_move."""
        key = (last_move.from_square, last_move.to_square)
        stored_counter = self.countermove_table.get(key)
        return stored_counter == move if stored_counter else False

    def update_countermove(self, move: chess.Move, last_move: chess.Move) -> None:
        """Update countermove table.

        Args:
            move: The move that refuted the opponent's move.
            last_move: The opponent's previous move.
        """
        if not self.config.use_countermove_heuristic:
            return

        if last_move:
            key = (last_move.from_square, last_move.to_square)
            self.countermove_table[key] = move

    # =========================================================================
    # Utilities
    # =========================================================================

    def _get_piece_value(self, square: int) -> int:
        """Get the value of a piece at a given square."""
        piece = self.board.piece_at(square)
        if piece:
            return int(PIECE_VALUES.get(piece.piece_type, 0))
        return 0

    def clear_killer_moves(self) -> None:
        """Clear all killer moves (e.g., at start of new search)."""
        if self.config.use_killer_moves:
            self.killer_moves.clear()

    def clear_history(self) -> None:
        """Clear history table."""
        if self.config.use_history_heuristic:
            self.history_table = [[0] * 64 for _ in range(64)]

    def clear_countermoves(self) -> None:
        """Clear countermove table."""
        if self.config.use_countermove_heuristic:
            self.countermove_table.clear()
