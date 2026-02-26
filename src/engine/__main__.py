"""Allow ``python -m engine`` invocation."""

from __future__ import annotations

import sys

from engine.uci import main as uci_main


def main() -> None:
    """Entry point for the chess engine."""
    uci_main()


if __name__ == "__main__":
    main()
