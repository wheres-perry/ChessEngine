"""Smoke tests for engine performance.

These tests act as 'canary in the coal mine' for major performance regressions
by verifying core operations complete within expected time thresholds.
"""

import time

import pytest

from engine._core import moray_core as core


def test_perft_speed_smoke():
    """Fail if Perft(5) takes longer than expected.

    This acts as a 'canary in the coal mine' for major performance regressions.
    """
    board = core.Board()

    # Depth 5 is usually ~4.8 million nodes.
    # On a modern laptop, this might take 0.5 - 2.0 seconds.
    # On a slow CI runner, it might take 4-5 seconds.
    depth = 5

    start = time.perf_counter()
    nodes = board.perft(depth)
    duration = time.perf_counter() - start

    nps = nodes / (duration + 0.0001)  # Avoid div by zero

    print(
        f"\n🔥 SMOKE BENCH: Depth {depth} | {nodes} nodes in {duration:.4f}s "
        f"({int(nps)} NPS)"
    )

    # THRESHOLD: Set this conservatively.
    # If your engine normally does 5M NPS, set this to 500k or 1M.
    # We only want to catch CATASTROPHIC failures here (e.g., 10x slowdown).
    min_nps = 500_000

    if nps < min_nps:
        pytest.fail(f"Performance Panic! NPS {int(nps)} is below threshold {min_nps}")
