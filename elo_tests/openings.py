"""Opening position loading and management.

This module provides functionality for loading chess opening positions
from EPD/FEN files or using a built-in default set.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_OPENINGS: list[str] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/3P4/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    "rnbqkb1r/ppp2ppp/3ppn2/8/2BP4/5N2/PPP2PPP/RNBQK2R b KQkq - 2 4",
    "r2q1rk1/ppp1bppp/2n1bn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 3 10",
    "2r2rk1/1bq1bppp/p2ppn2/1pn5/3NP3/1PN1BP2/PBQ2P1P/2RR2K1 w - - 1 15",
]
"""Default opening positions as FEN strings."""


def _line_to_fen(line: str) -> str:
    """Convert an EPD/FEN line to a valid FEN string.

    Args:
        line: Raw line from an openings file.

    Returns:
        A valid FEN string (first 6 fields).

    """
    return " ".join(line.strip().split()[:6])


def load_opening_fens(path: str | None) -> list[str]:
    """Load opening positions from a file or return defaults.

    Args:
        path: Path to an EPD/FEN file, or None for defaults.

    Returns:
        A list of FEN strings for opening positions.

    """
    if not path:
        return list(DEFAULT_OPENINGS)

    file_path = Path(path)
    if not file_path.exists():
        return list(DEFAULT_OPENINGS)

    openings: list[str] = []
    for raw in file_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        openings.append(_line_to_fen(line))

    return openings or list(DEFAULT_OPENINGS)
