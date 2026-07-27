"""Golden file regression tests for engine consistency."""

import pytest

from engine._core import moray_core as core


def load_fens():
    """Load FEN positions from the test data file."""
    try:
        with open("tests/parity/data/fens.txt") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []


class TestGoldenParity:
    """Golden file regression tests for engine consistency."""

    def test_engine_consistency(self, data_regression):
        """Run the engine on a suite of FENs and compare results to the 'Golden' file.

        If logic changes (e.g., node count changes), this test FAILS and shows the diff.
        """
        fens = load_fens()
        if not fens:
            pytest.skip("No FENs found in tests/parity/data/fens.txt")

        results = {}
        board = core.Board()

        for fen in fens:
            board.set_fen(fen)
            # Run a fixed depth search (e.g., depth 4).
            # This is deterministic and creates a "fingerprint" of your engine's logic.
            nodes = board.perft(4)

            # Store meaningful metrics for this FEN
            results[fen] = {
                "nodes": nodes,
                # You can add "score" or "best_move" here later
            }

        # This compares 'results' against 'test_engine_consistency.yml'
        # To update the golden file, run pytest with: --force-regen
        data_regression.check(results)
