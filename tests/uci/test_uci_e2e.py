"""End-to-end UCI protocol verification tests.

Spawns the chess engine as a subprocess and verifies UCI protocol compliance
by sending commands and checking expected responses.
"""

import subprocess
import sys
import time
from collections.abc import Generator

import pytest

# Command to launch the engine directly via the current python executable
_ENGINE_CMD = [sys.executable, "-m", "engine"]

# Starting position with e2e4 already played (used for position fen test)
_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4"


@pytest.fixture
def engine_process() -> Generator[subprocess.Popen[str], None, None]:
    """Spawn the engine subprocess with line buffering for real-time I/O."""
    process = subprocess.Popen(  # noqa: S603
        _ENGINE_CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1,
    )

    yield process

    # Cleanup after test
    if process.poll() is None:
        if process.stdin:
            process.stdin.write("quit\n")
            process.stdin.flush()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait()


def _send(process: subprocess.Popen[str], cmd: str) -> None:
    """Send a command to the engine."""
    assert process.stdin is not None
    process.stdin.write(cmd + "\n")
    process.stdin.flush()


def _expect(process: subprocess.Popen[str], pattern: str, timeout: float = 2.0) -> bool:
    """Wait for a pattern to appear in engine output."""
    assert process.stdout is not None
    start = time.time()
    while time.time() - start < timeout:
        line = process.stdout.readline()
        if not line:
            break
        line = line.strip()
        if pattern in line:
            return True
    return False


def test_uci_handshake(engine_process: subprocess.Popen[str]) -> None:
    """Test engine responds to 'uci' with 'id name', 'id author', and 'uciok'."""
    _send(engine_process, "uci")
    assert _expect(engine_process, "id name ChessEngine")
    assert _expect(engine_process, "id author")
    assert _expect(engine_process, "uciok")


def test_isready(engine_process: subprocess.Popen[str]) -> None:
    """Test engine responds to 'isready' with 'readyok'."""
    _send(engine_process, "isready")
    assert _expect(engine_process, "readyok")


def test_go_startpos(engine_process: subprocess.Popen[str]) -> None:
    """Test engine can search and return a move from the starting position."""
    _send(engine_process, "position startpos")
    _send(engine_process, "go depth 1")
    assert _expect(engine_process, "bestmove")


def test_go_fen(engine_process: subprocess.Popen[str]) -> None:
    """Test engine can search and return a move from a custom FEN."""
    _send(engine_process, f"position fen {_START_FEN}")
    _send(engine_process, "go depth 1")
    assert _expect(engine_process, "bestmove")
