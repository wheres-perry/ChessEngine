"""Re-exports the C++ Zobrist hashing class for use from Python search code."""

from engine._core import moray_core as chess

# The real implementation lives in the compiled C++ extension.
Zobrist = chess.Zobrist
