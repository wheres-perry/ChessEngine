# pylint: disable=all
# mypy: ignore-errors
# ruff: noqa
# fmt: off
# This is a benchmark file where performance testing is the priority

import statistics
import time
from itertools import islice

import chess  # The python-chess library
import pytest

# Editable install exposes the extension as engine._core.chess_engine_core
from engine._core import chess_engine_core as core  # Your custom library

# ==============================================================================
# Configuration
# ==============================================================================

FEN_FILE_PATH = "/workspace/benchmarks/fens.txt"
LINES_TO_BENCHMARK = 100_000
BENCHMARK_ROUNDS = 20  # Number of rounds for manual testing

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
# Pytest Integration (with multiple rounds)
# ==============================================================================


@pytest.fixture(scope="session")
def fen_strings():
    """Pytest fixture to provide FEN strings to the benchmark tests."""
    return load_fens(FEN_FILE_PATH, LINES_TO_BENCHMARK)


def test_benchmark_python_chess(benchmark, fen_strings):
    """Benchmarks the standard 'python-chess' library with multiple rounds."""
    benchmark.pedantic(
        process_fens_library, args=(fen_strings,), rounds=BENCHMARK_ROUNDS
    )


def test_benchmark_custom_engine(benchmark, fen_strings):
    """Benchmarks your custom core engine with multiple rounds."""
    benchmark.pedantic(
        process_fens_custom, args=(fen_strings,), rounds=BENCHMARK_ROUNDS
    )


# ==============================================================================
# Direct Script Execution for Quick Manual Check
# ==============================================================================


def run_multiple_rounds(func, fens, rounds=BENCHMARK_ROUNDS):
    """Runs a function multiple times and returns timing statistics."""
    times = []
    for round_num in range(rounds):
        print(f"    Round {round_num + 1}/{rounds}...")
        start_time = time.perf_counter()
        func(fens)
        end_time = time.perf_counter()
        duration = end_time - start_time
        times.append(duration)

    return times


def print_statistics(times, num_fens, library_name):
    """Prints detailed statistics for the timing results."""
    mean_time = statistics.mean(times)
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0

    print(f"\n{library_name} Results ({len(times)} rounds):")
    print(f"  -> Mean time: {mean_time:.4f} seconds")
    print(f"  -> Median time: {median_time:.4f} seconds")
    print(f"  -> Min time: {min_time:.4f} seconds")
    print(f"  -> Max time: {max_time:.4f} seconds")
    print(f"  -> Std deviation: {std_dev:.4f} seconds")
    print(f"  -> Mean time per FEN: {mean_time / num_fens:.3e} seconds")
    print(f"  -> All times: {[f'{t:.4f}' for t in times]}")


def main():
    """Main function to run a quick, non-benchmark comparison with multiple rounds."""
    print("--- Direct Script Execution (Manual Timing with Multiple Rounds) ---")
    fens = load_fens(FEN_FILE_PATH, LINES_TO_BENCHMARK)
    num_fens = len(fens)
    print(f"Loaded {num_fens:,} FENs for comparison.")
    print(f"Running {BENCHMARK_ROUNDS} rounds for each library.\n")

    # Time python-chess
    print("Running 'python-chess'...")
    lib_times = run_multiple_rounds(process_fens_library, fens)
    print_statistics(lib_times, num_fens, "'python-chess'")

    # Time custom engine
    print("\nRunning custom engine...")
    custom_times = run_multiple_rounds(process_fens_custom, fens)
    print_statistics(custom_times, num_fens, "Custom Engine")

    # Compare results
    lib_mean = statistics.mean(lib_times)
    custom_mean = statistics.mean(custom_times)
    speedup = lib_mean / custom_mean if custom_mean > 0 else float("inf")

    print("\n--- Comparison ---")
    print(
        f"Custom engine is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than python-chess"
    )
    print(f"Performance difference: {abs(speedup - 1) * 100:.1f}%")

    print("\n--- To run a proper benchmark with pytest-benchmark, use: pytest ---")


if __name__ == "__main__":
    main()
