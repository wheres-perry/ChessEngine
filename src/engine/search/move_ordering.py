"""Move ordering heuristics for negamax search."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from engine._core import chess_engine_core as chess

if TYPE_CHECKING:
    from engine.config import SearchConfig


class MoveSorter:
    """Scores and sorts moves using configurable heuristic tiers."""

    HASH_MOVE_SCORE: int = 100_000_000
    TACTICAL_BASE: int = 10_000_000
    KILLER_BASE: int = 1_000_000
    COUNTERMOVE_SCORE: int = 850_000

    PIECE_VALUES_CP: ClassVar[dict[int, int]] = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20_000,
    }

    def __init__(self, config: SearchConfig) -> None:
        """Initialize empty killer, history, and countermove tables.

        Args:
            config: The search configuration containing move ordering settings.

        """
        self.config = config
        self.killer_moves: dict[int, list[chess.Move]] = {}
        self.history_table: dict[tuple[int, int, int], int] = {}
        self.countermove_table: dict[tuple[int, int, int], chess.Move] = {}

    def reset(self, clear_history: bool = True, clear_killers: bool = True) -> None:
        """Clear heuristic tables; each flag controls one table independently."""
        if clear_killers:
            self.killer_moves.clear()
        if clear_history:
            self.history_table.clear()
            self.countermove_table.clear()

    def sort_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        ply: int,
        hash_move: chess.Move | None,
        previous_move: chess.Move | None,
    ) -> list[chess.Move]:
        """Score and sort all moves in descending priority order.

        Args:
            board: The current board state.
            moves: The list of moves to sort.
            ply: The current ply (depth) in the search.
            hash_move: The hash move from the transposition table, if any.
            previous_move: The previous move made, for countermove heuristic.

        Returns:
            The sorted list of moves in descending priority order.

        """
        if not self.config.use_move_ordering or len(moves) <= 1:
            return moves

        scored_moves = [
            (
                self._score_move(
                    board=board,
                    move=move,
                    ply=ply,
                    hash_move=hash_move,
                    previous_move=previous_move,
                ),
                move,
            )
            for move in moves
        ]
        scored_moves.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored_moves]

    def sort_tactical(
        self, board: chess.Board, moves: list[chess.Move]
    ) -> list[chess.Move]:
        """Sort captures/promotions by MVV-LVA + SEE; used in quiescence search.

        Args:
            board: The current board state.
            moves: The list of tactical moves (captures/promotions) to sort.

        Returns:
            The sorted list of tactical moves in descending priority order.

        """
        scored_moves = [
            (self._score_tactical_move(board, move), move) for move in moves
        ]
        scored_moves.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored_moves]

    def _score_move(
        self,
        board: chess.Board,
        move: chess.Move,
        ply: int,
        hash_move: chess.Move | None,
        previous_move: chess.Move | None,
    ) -> int:
        """Assign an integer priority score to a single move.

        Args:
            board: The current board state.
            move: The move to score.
            ply: The current ply (depth) in the search.
            hash_move: The hash move from the transposition table, if any.
            previous_move: The previous move made, for countermove heuristic.

        Returns:
            The integer priority score for the move.

        """
        if (
            self.config.use_hash_move_ordering
            and hash_move is not None
            and move == hash_move
        ):
            return self.HASH_MOVE_SCORE

        if board.is_capture(move) or self._is_promotion(move):
            return self._score_tactical_move(board, move)

        if self.config.use_killer_moves:
            killers = self.killer_moves.get(ply, [])
            if move in killers:
                slot = killers.index(move)
                return self.KILLER_BASE - (slot * 1024)

        if self.config.use_countermove_heuristic and previous_move is not None:
            key = self._move_key(previous_move)
            countermove = self.countermove_table.get(key)
            if countermove is not None and countermove == move:
                return self.COUNTERMOVE_SCORE

        if self.config.use_history_heuristic:
            history = self.history_table.get(self._move_key(move), 0)
            return min(history, int(self.config.history_max_score))

        return 0

    def _score_tactical_move(self, board: chess.Board, move: chess.Move) -> int:
        """Score a capture or promotion using MVV-LVA, promotions, and SEE.

        Args:
            board: The current board state.
            move: The tactical move to score.

        Returns:
            The integer priority score for the tactical move.

        """
        score = self.TACTICAL_BASE
        if self.config.use_mvv_lva and board.is_capture(move):
            score += self._mvv_lva(board, move)
        if self._is_promotion(move):
            score += self.PIECE_VALUES_CP.get(move.promotion, 0)
        if self.config.use_see_ordering and board.is_capture(move):
            see_value = self.see(board, move)
            if see_value < self.config.see_capture_threshold:
                score -= 50_000
            else:
                score += min(see_value, 5_000)
        return score

    def _mvv_lva(self, board: chess.Board, move: chess.Move) -> int:
        """Return the MVV-LVA bonus for a capture move.

        MVV-LVA (Most Valuable Victim - Least Valuable Aggressor) prioritizes
        capturing high-value pieces with low-value pieces.

        Args:
            board: The current board state.
            move: The capture move to evaluate.

        Returns:
            The MVV-LVA bonus score (victim_value * 10 - attacker_value).

        """
        victim_piece = board.piece_at(move.to_square)
        attacker_piece = board.piece_at(move.from_square)

        victim_value = (
            self.PIECE_VALUES_CP[chess.PAWN]
            if victim_piece is None and board.is_en_passant(move)
            else self.PIECE_VALUES_CP.get(victim_piece.piece_type, 0)
            if victim_piece is not None
            else 0
        )
        attacker_value = (
            self.PIECE_VALUES_CP.get(
                attacker_piece.piece_type, self.PIECE_VALUES_CP[chess.PAWN]
            )
            if attacker_piece is not None
            else self.PIECE_VALUES_CP[chess.PAWN]
        )
        return victim_value * 10 - attacker_value

    def see(self, board: chess.Board, move: chess.Move) -> int:
        """Calculate a simplified SEE approximation for pruning/ordering decisions.

        SEE (Static Exchange Evaluation) estimates the material gain/loss from
        a capture sequence.

        Args:
            board: The current board state.
            move: The capture move to evaluate.

        Returns:
            The estimated SEE value (victim value - attacker value).

        """
        if not board.is_capture(move):
            return 0
        victim_piece = board.piece_at(move.to_square)
        attacker_piece = board.piece_at(move.from_square)
        victim_value = (
            self.PIECE_VALUES_CP[chess.PAWN]
            if victim_piece is None and board.is_en_passant(move)
            else self.PIECE_VALUES_CP.get(victim_piece.piece_type, 0)
            if victim_piece is not None
            else 0
        )
        attacker_value = (
            self.PIECE_VALUES_CP.get(
                attacker_piece.piece_type, self.PIECE_VALUES_CP[chess.PAWN]
            )
            if attacker_piece is not None
            else self.PIECE_VALUES_CP[chess.PAWN]
        )
        return victim_value - attacker_value

    def on_beta_cutoff(
        self,
        move: chess.Move,
        ply: int,
        depth: int,
        previous_move: chess.Move | None,
        is_tactical: bool,
    ) -> None:
        """Update killers, history, and countermove tables after a beta cutoff.

        Args:
            move: The move that caused the beta cutoff.
            ply: The current ply (depth) where the cutoff occurred.
            depth: The remaining search depth.
            previous_move: The previous move made, for countermove heuristic.
            is_tactical: Whether the cutoff move was tactical (capture/promotion).

        """
        if is_tactical:
            return

        if self.config.use_killer_moves:
            killers = self.killer_moves.setdefault(ply, [])
            if move in killers:
                return
            killers.insert(0, move)
            max_killers = max(1, self.config.killer_slots_per_ply)
            if len(killers) > max_killers:
                del killers[max_killers:]

        if self.config.use_history_heuristic:
            key = self._move_key(move)
            bonus = depth * depth
            current = self.history_table.get(key, 0)
            self.history_table[key] = min(
                current + bonus,
                self.config.history_max_score,
            )

        if self.config.use_countermove_heuristic and previous_move is not None:
            self.countermove_table[self._move_key(previous_move)] = move

    def history_saturation(self) -> float:
        """Return the history table saturation as a percentage (0-100).

        100 means fully saturated (all entries at max score).

        Returns:
            The history table saturation percentage.

        """
        if not self.config.use_history_heuristic or not self.history_table:
            return 0.0
        max_score = float(self.config.history_max_score)
        avg_score = sum(self.history_table.values()) / len(self.history_table)
        return min(100.0, (avg_score / max_score) * 100.0)

    @staticmethod
    def _is_promotion(move: chess.Move) -> bool:
        """Check if a move is a promotion.

        Args:
            move: The move to check.

        Returns:
            True if the move is a promotion, False otherwise.

        """
        return int(move.promotion) != 0

    @staticmethod
    def _move_key(move: chess.Move) -> tuple[int, int, int]:
        """Generate a hashable key for a move.

        Args:
            move: The move to generate a key for.

        Returns:
            A tuple of (from_square, to_square, promotion) representing the move.

        """
        return move.from_square, move.to_square, int(move.promotion)
