"""Chess puzzle test fixtures and data loading."""

from __future__ import annotations

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


def load_fen_file(name: str) -> list[tuple[str, str]]:
    """Load FEN puzzles from a data file.

    Each line: ``<FEN> ; <expected best move in UCI>``
    Blank lines and ``#`` comments are skipped.

    Returns list of ``(fen, expected_move)`` tuples.
    """
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"Data file not found: {path}")
    puzzles: list[tuple[str, str]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fen, move = line.split(";", 1)
        puzzles.append((fen.strip(), move.strip()))
    return puzzles


@pytest.fixture
def mate_in_1_puzzles() -> list[tuple[str, str]]:
    """Mate-in-1 puzzles from ``data/mate_in_1.fen``."""
    return load_fen_file("mate_in_1.fen")
