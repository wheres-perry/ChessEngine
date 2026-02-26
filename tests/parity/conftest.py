"""Parity test fixtures and data loading."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_fens() -> list[str]:
    """Load FEN positions from ``data/fens.txt``."""
    path = DATA_DIR / "fens.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]
