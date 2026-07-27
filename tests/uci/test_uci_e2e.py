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
    assert _expect(engine_process, "id name Moray")
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


def test_go_reported_issue_position_with_time_control(
    engine_process: subprocess.Popen[str],
) -> None:
    """Test engine handles reported move sequence and time control."""
    position_cmd = (
        "position startpos moves "
        "e2e4 b8c6 b1c3 g8f6 d2d4 g7g6 f1b5 c6b4 a2a3 b4c6 "
        "b5c6 d7c6 g1f3 c8g4 e1g1 g4f3 d1f3 d8d4 c1e3 d4c4 "
        "a1d1 a7a5 e4e5 f6g4 d1d4 c4f1 g1f1"
    )
    _send(engine_process, position_cmd)
    _send(engine_process, "go wtime 120000 btime 4400 winc 0 binc 0")

    assert engine_process.stdout is not None
    start = time.time()
    best_move_line = ""
    while time.time() - start < 3.0:
        line = engine_process.stdout.readline()
        if not line:
            break
        if "bestmove" in line:
            best_move_line = line.strip()
            break

    assert best_move_line != ""
    assert "bestmove 0000" not in best_move_line
    assert "bestmove g4h2" in best_move_line
