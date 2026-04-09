"""Syzygy endgame tablebase probing.

Wraps python-chess's ``chess.syzygy.Tablebase`` via a C++ ``CppSyzygyProber``
that performs fast piece counting (hardware popcount) and delegates the
actual WDL/DTZ lookups back to Python through stored callbacks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chess as python_chess
import chess.syzygy

from engine._core import chess_engine_core as core

logger = logging.getLogger(__name__)

# Maximum piece count for tablebase probing (3-4-5 piece tables).
MAX_PIECES = 5


def _make_probe_callback(
    tb: chess.syzygy.Tablebase,
    probe_method: str,
) -> object:
    """Return a callback ``(fen, use_50mr) -> int|None`` for *probe_method*.

    The callback creates a python-chess Board from the FEN, optionally
    zeroes the halfmove clock (when *use_50mr* is False), and calls the
    appropriate tablebase probe method.
    """

    def _callback(fen: str, use_50_move_rule: bool) -> int | None:
        pc_board = python_chess.Board(fen)
        try:
            if use_50_move_rule:
                return getattr(tb, probe_method)(pc_board)
            saved_hmc = pc_board.halfmove_clock
            pc_board.halfmove_clock = 0
            result: int = getattr(tb, probe_method)(pc_board)
            pc_board.halfmove_clock = saved_hmc
            return result
        except KeyError:
            return None

    return _callback


class SyzygyProber:
    """Probe Syzygy endgame tablebases for WDL and DTZ results."""

    def __init__(self, path: str, *, use_50_move_rule: bool = True) -> None:
        """Open Syzygy tables from *path*.

        Args:
            path: Directory containing ``.rtbw`` / ``.rtbz`` files.
            use_50_move_rule: When True, tablebase results respect the
                50-move drawing rule.  When False, positions that would
                be drawn under the 50-move rule are still reported as
                wins/losses (useful for finding forced mates that exceed
                50 moves without a pawn push or capture).

        """
        self._use_50_move_rule = use_50_move_rule
        tb = chess.syzygy.Tablebase()
        tb_path = Path(path)
        if tb_path.is_dir():
            tb.add_directory(str(tb_path))
            logger.debug("Syzygy tables loaded from %s", tb_path)
        else:
            logger.warning("Syzygy path %s is not a directory", tb_path)

        # Build Python callbacks that the C++ prober will invoke.
        wdl_cb = _make_probe_callback(tb, "probe_wdl")
        dtz_cb = _make_probe_callback(tb, "probe_dtz")

        # Keep a reference to the tablebase so it isn't garbage-collected
        # while the callbacks still reference it.
        self._tb = tb

        self._cpp = core.CppSyzygyProber(path, use_50_move_rule, wdl_cb, dtz_cb)

    @staticmethod
    def piece_count(board: core.Board) -> int:
        """Return the total number of pieces on the board."""
        return core.CppSyzygyProber.piece_count(board)

    def probe_wdl(self, board: core.Board) -> int | None:
        """Probe the WDL (Win/Draw/Loss) table.

        Returns:
            2 = win, 1 = cursed win, 0 = draw, -1 = blessed loss,
            -2 = loss.  None if the position is not in the tablebase.

        """
        return self._cpp.probe_wdl(board)

    def probe_dtz(self, board: core.Board) -> int | None:
        """Probe the DTZ (Distance To Zeroing) table.

        Returns:
            Positive = side to move wins in N plies, negative = loses,
            zero = draw.  None if not in tablebase.

        """
        return self._cpp.probe_dtz(board)

    def close(self) -> None:
        """Release tablebase file handles."""
        self._tb.close()
