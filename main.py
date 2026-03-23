"""Entry point for running the chess engine directly from the root directory."""

import sys
from pathlib import Path

# Add src to the Python path so engine can be imported directly
src_path = Path(__file__).resolve().parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from engine.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
