import time
from itertools import islice

import chess  # The standard library being tested
import pytest

from src.engine._core import chess_engine_core as core  # Your custom library

# ==============================================================================
# Configuration
# ==============================================================================

FEN_FILE_PATH = "/workspace/benchmarks/fens.txt"
LINES_TO_BENCHMARK = 1_000_000

# ==============================================================================
# Data Loading
# ==============================================================================


def load_fens(path, limit):
    """Loads a specified number of lines from the FEN file."""
    with open(path) as f:
        lines = islice(f, limit)
        return [line.strip() for line in lines if line.strip()]


# ==============================================================================
# Processing Functions for Each Library
# ==============================================================================


def process_fens_library(fens):
    """Processes FENs using the standard 'python-chess' library."""
    for fen in fens:
        board = chess.Board(fen)
        # list() is necessary to consume the generator and do the actual work.
        _ = list(board.legal_moves)


def process_fens_custom(fens):
    """Processes FENs using your custom core engine."""
    for fen in fens:
        board = core.Board.from_fen(fen)
        board.generate_legal_moves()


# ==============================================================================
# Pytest Integration
# ==============================================================================


@pytest.fixture(scope="session")
def fen_strings():
    """Pytest fixture to provide FEN strings to the benchmark tests."""
    return load_fens(FEN_FILE_PATH, LINES_TO_BENCHMARK)


def test_benchmark_python_chess(benchmark, fen_strings):
    """Benchmarks the standard 'python-chess' library."""
    benchmark(process_fens_library, fen_strings)


def test_benchmark_custom_engine(benchmark, fen_strings):
    """Benchmarks your custom core engine."""
    benchmark(process_fens_custom, fen_strings)


# ==============================================================================
# Direct Script Execution for Quick Manual Check
# ==============================================================================


def main():
    """Main function to run a quick, non-benchmark comparison."""
    print("--- Direct Script Execution (Manual Timing) ---")
    fens = load_fens(FEN_FILE_PATH, LINES_TO_BENCHMARK)
    num_fens = len(fens)
    print(f"Loaded {num_fens:,} FENs for comparison.\n")

    # Time python-chess
    print("Running 'python-chess'...")
    start_time_lib = time.perf_counter()
    process_fens_library(fens)
    end_time_lib = time.perf_counter()
    duration_lib = end_time_lib - start_time_lib
    avg_time_lib = duration_lib / num_fens
    print(f"  -> Total time: {duration_lib:.4f} seconds.")
    print(f"  -> Avg time per FEN: {avg_time_lib:.3e} seconds.")

    # Time custom engine
    print("\nRunning custom engine...")
    start_time_custom = time.perf_counter()
    process_fens_custom(fens)
    end_time_custom = time.perf_counter()
    duration_custom = end_time_custom - start_time_custom
    avg_time_custom = duration_custom / num_fens
    print(f"  -> Total time: {duration_custom:.4f} seconds.")
    print(f"  -> Avg time per FEN: {avg_time_custom:.3e} seconds.")

    print("\n--- To run a proper benchmark, use: pytest ---")


if __name__ == "__main__":
    main()
