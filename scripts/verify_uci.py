"""End-to-end UCI protocol verification script.

Spawns the chess engine as a subprocess and verifies UCI protocol compliance
by sending commands and checking expected responses.
"""

import subprocess
import sys
import time

# Command to launch the engine via uv
_ENGINE_CMD = ["uv", "run", "python", "-m", "engine"]

# Starting position with e2e4 already played (used for position fen test)
_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4"


def _start_engine():
    """Spawn the engine subprocess with line buffering for real-time I/O.

    Returns:
        A Popen object representing the engine process.
    """
    return subprocess.Popen(  # noqa: S603
        _ENGINE_CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1,
    )


def _send(process, cmd):
    """Send a command to the engine.

    Args:
        process: The engine subprocess.
        cmd: The UCI command string to send.
    """
    print(f"> {cmd}")
    process.stdin.write(cmd + "\n")
    process.stdin.flush()


def _expect(process, pattern, timeout=2.0):
    """Wait for a pattern to appear in engine output.

    Args:
        process: The engine subprocess.
        pattern: The string pattern to search for.
        timeout: Maximum time to wait in seconds.

    Returns:
        True if pattern found, False otherwise.
    """
    start = time.time()
    while time.time() - start < timeout:
        line = process.stdout.readline()
        if not line:
            break
        line = line.strip()
        if pattern in line:
            print(f"Verified: {pattern}")
            return True
    print(f"Failed to find: {pattern}")
    return False


def _run_checks(process):
    """Execute the sequence of UCI checks.

    Args:
        process: The engine subprocess.

    Returns:
        True if all checks pass, False otherwise.
    """
    _send(process, "uci")
    if not _expect(process, "id name ChessEngine"):
        return False
    if not _expect(process, "uciok"):
        return False

    _send(process, "isready")
    if not _expect(process, "readyok"):
        return False

    _send(process, "position startpos")
    _send(process, "go depth 1")
    if not _expect(process, "bestmove"):
        return False

    _send(process, f"position fen {_START_FEN}")
    _send(process, "go depth 1")
    return _expect(process, "bestmove")


def verify_uci():
    """Run the full UCI verification sequence.

    Returns:
        True if verification passes, False otherwise.
    """
    process = _start_engine()
    ok = _run_checks(process)

    if not ok:
        process.terminate()
        return False

    _send(process, "quit")
    process.wait(timeout=2)
    if process.returncode is not None:
        print("Engine exited successfully.")
    else:
        print("Engine did not exit.")
        process.terminate()
        return False

    return True


if __name__ == "__main__":
    if verify_uci():
        print("UCI Verification Passed!")
        sys.exit(0)
    else:
        print("UCI Verification Failed!")
        sys.exit(1)
