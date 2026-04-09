"""Move ordering heuristics for negamax search (C++ backed)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from engine._core import chess_engine_core as chess

if TYPE_CHECKING:
    from engine.config import SearchConfig


class _KillerMovesProxy:
    """Dict-like proxy for C++ killer moves, supporting [ply] and .get()."""

    __slots__ = ("_sorter",)

    def __init__(self, sorter: chess.MoveSorter) -> None:
        self._sorter = sorter

    def __getitem__(self, ply: int) -> list[chess.Move]:
        return list(self._sorter.get_killers(ply))

    def get(
        self, ply: int, default: list[chess.Move] | None = None
    ) -> list[chess.Move]:
        """Return killer moves for *ply*, or *default* if empty."""
        killers = list(self._sorter.get_killers(ply))
        if killers:
            return killers
        return default if default is not None else []

    def clear(self) -> None:
        """Clear all killer moves via reset."""
        self._sorter.reset(clear_history=False, clear_killers=True)

    def setdefault(
        self, ply: int, default: list[chess.Move] | None = None
    ) -> list[chess.Move]:
        """Return killer moves for *ply*; unlike a real dict, does not insert."""
        killers = list(self._sorter.get_killers(ply))
        if killers:
            return killers
        return default if default is not None else []


class _HistoryTableProxy:
    """Dict-like proxy for C++ history table, supporting [(f,t,p)] and .get()."""

    __slots__ = ("_sorter",)

    def __init__(self, sorter: chess.MoveSorter) -> None:
        self._sorter = sorter

    def __getitem__(self, key: tuple[int, int, int]) -> int:
        from_sq, to_sq, promo = key
        val = self._sorter.get_history(from_sq, to_sq, promo)
        if val == 0:
            # Check if key actually exists in the table
            table = self._sorter.get_history_table()
            if key not in table:
                raise KeyError(key)
        return val

    def get(self, key: tuple[int, int, int], default: int = 0) -> int:
        """Return history score for *key*, or *default* if not present."""
        table = self._sorter.get_history_table()
        if key in table:
            return table[key]
        return default

    def clear(self) -> None:
        """Clear history and countermove tables."""
        self._sorter.reset(clear_history=True, clear_killers=False)

    def values(self) -> list[int]:
        """Return all history values."""
        return list(self._sorter.get_history_table().values())

    def __len__(self) -> int:
        return len(self._sorter.get_history_table())

    def __bool__(self) -> bool:
        return len(self._sorter.get_history_table()) > 0


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
        """Initialize C++ MoveSorter from a SearchConfig.

        Args:
            config: The search configuration containing move ordering settings.

        """
        self.config = config

        cpp_cfg = chess.MoveSorterConfig()
        cpp_cfg.use_move_ordering = config.use_move_ordering
        cpp_cfg.use_mvv_lva = config.use_mvv_lva
        cpp_cfg.use_history_heuristic = config.use_history_heuristic
        cpp_cfg.use_countermove_heuristic = config.use_countermove_heuristic
        cpp_cfg.use_see_ordering = config.use_see_ordering
        cpp_cfg.use_killer_moves = config.use_killer_moves
        cpp_cfg.use_hash_move_ordering = config.use_hash_move_ordering
        cpp_cfg.history_max_score = config.history_max_score
        cpp_cfg.killer_slots_per_ply = config.killer_slots_per_ply
        cpp_cfg.see_capture_threshold = config.see_capture_threshold

        self._cpp = chess.MoveSorter(cpp_cfg)
        self.killer_moves: _KillerMovesProxy = _KillerMovesProxy(self._cpp)
        self.history_table: _HistoryTableProxy = _HistoryTableProxy(self._cpp)
        self.countermove_table: dict[tuple[int, int, int], chess.Move] = {}

    def reset(self, clear_history: bool = True, clear_killers: bool = True) -> None:
        """Clear heuristic tables; each flag controls one table independently."""
        self._cpp.reset(clear_history=clear_history, clear_killers=clear_killers)

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
        return self._cpp.sort_moves(
            board=board,
            moves=moves,
            ply=ply,
            hash_move=hash_move,
            previous_move=previous_move,
        )

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
        return self._cpp.sort_tactical(board=board, moves=moves)

    def see(self, board: chess.Board, move: chess.Move) -> int:
        """Calculate a precise SEE (Static Exchange Evaluation) for pruning/ordering.

        Args:
            board: The current board state.
            move: The initial capture move to evaluate.

        Returns:
            The estimated SEE value (positive is good, negative is bad).

        """
        return self._cpp.see(board=board, move=move)

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
        self._cpp.on_beta_cutoff(
            move=move,
            ply=ply,
            depth=depth,
            previous_move=previous_move,
            is_tactical=is_tactical,
        )

    def history_saturation(self) -> float:
        """Return the history table saturation as a percentage (0-100).

        100 means fully saturated (all entries at max score).

        Returns:
            The history table saturation percentage.

        """
        return self._cpp.history_saturation()
