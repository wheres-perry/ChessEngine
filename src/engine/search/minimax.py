"""Thin Python wrapper around the C++ negamax searcher.

The real search loop lives in ``engine._core.chess_engine_core.CppMinimax``.
This module preserves the legacy ``Minimax`` import path and exposes the
attributes the existing test suite and factory layer touch directly
(``stats``, ``tt``, ``move_sorter``, ``zobrist``, ``start_time``,
``time_up``, ``_check_time_limit``…).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from engine._core import chess_engine_core as chess
from engine.search.move_ordering import MoveSorter
from engine.search.stats import SearchStats
from engine.search.transposition_table import TranspositionTable
from engine.search.zobrist import Zobrist

if TYPE_CHECKING:
    from engine.config import EngineConfig
    from engine.evaluators import Evaluator

# Fields that exist on both the native MinimaxStats struct and the Python
# SearchStats dataclass.  We copy these from C++ into Python after every
# search so tests/UCI code can observe idiomatic Python dataclass fields.
_SHARED_STATS_FIELDS: tuple[str, ...] = (
    "nodes",
    "depth",
    "seldepth",
    "tt_hits",
    "hashfull",
    "beta_cutoffs",
    "first_move_cuts",
    "killer_cuts",
    "history_cuts",
    "qsearch_nodes",
    "null_move_cuts",
    "pvs_researches",
    "lmr_researches",
    "qs_see_pruning",
    "qs_delta_pruning",
    "check_extensions",
    "iid_searches",
    "root_move_changes",
    "history_saturation",
    "score",
)


class Minimax:
    """Config-driven negamax searcher backed by the C++ implementation."""

    NEG_INF = float("-inf")
    POS_INF = float("inf")
    MATE_SCORE = 100_000
    TIME_CHECK_INTERVAL = 2048

    def __init__(
        self,
        board: chess.Board,
        evaluator: Evaluator,
        config: EngineConfig,
    ) -> None:
        """Build the underlying C++ search and all its dependencies."""
        self.board = board
        self.evaluator = evaluator
        self.config = config
        self.search_cfg = config.search

        self.stats = SearchStats()
        self.node_count = 0
        self.time_up = False
        self.start_time: float | None = None

        self.zobrist: Zobrist | None = None
        self.tt: TranspositionTable | None = None
        if self.search_cfg.use_transposition_table:
            self.zobrist = Zobrist()
            self.tt = TranspositionTable(self.search_cfg)
            self.zobrist.hash_board(self.board)

        self.move_sorter: MoveSorter | None = None
        if self.search_cfg.use_move_ordering:
            self.move_sorter = MoveSorter(self.search_cfg)

        self.root_best_move: chess.Move | None = None

        self._cpp = chess.CppMinimax(
            board,
            evaluator,
            self.tt,
            self.move_sorter,
            self.zobrist,
            self.search_cfg,
        )

    # ── Public API ────────────────────────────────────────────────
    def reset_state(
        self,
        clear_tt: bool = True,
        clear_history: bool = True,
        clear_killers: bool = True,
    ) -> None:
        """Reset search state; optionally preserve TT, history, and killer tables."""
        self._cpp.reset_state(
            clear_tt=clear_tt,
            clear_history=clear_history,
            clear_killers=clear_killers,
        )
        self.stats.reset()
        self.node_count = 0
        self.root_best_move = None

    def find_best_move(
        self,
        depth: int | None = None,
    ) -> tuple[float | None, chess.Move | None]:
        """Run IDDFS up to *depth*.

        Returns the best (score, move) pair from White's perspective.
        """
        target_depth = max(1, depth if depth is not None else self.config.search_depth)

        self.start_time = time.time()
        self.time_up = False
        self.root_best_move = None

        if self.zobrist is not None:
            self.zobrist.hash_board(self.board)

        score, move = self._cpp.find_best_move(target_depth)
        self._sync_stats_from_cpp()

        self.time_up = bool(self._cpp.time_up)
        self.root_best_move = self._cpp.root_best_move
        self.node_count = int(self.stats.nodes)
        return score, move

    def find_top_move(self, depth: int = 1) -> tuple[float | None, chess.Move | None]:
        """Backward-compatible alias for previous API."""
        return self.find_best_move(depth)

    # ── Test/debug helpers ───────────────────────────────────────
    def _check_time_limit(self) -> bool:
        """Check if the search has exceeded the configured time limit."""
        max_time = self.search_cfg.max_time
        if max_time is None or self.start_time is None:
            return False
        if time.time() - self.start_time >= max_time:
            self.time_up = True
            return True
        return False

    # ── Internal helpers ─────────────────────────────────────────
    def _sync_stats_from_cpp(self) -> None:
        """Copy C++ MinimaxStats fields into the Python SearchStats dataclass."""
        cpp_stats = self._cpp.stats
        for field in _SHARED_STATS_FIELDS:
            setattr(self.stats, field, getattr(cpp_stats, field))
