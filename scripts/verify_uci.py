"""End-to-end UCI protocol verification script."""

import subprocess
import sys
import time

_ENGINE_CMD = ["uv", "run", "python", "-m", "engine"]

_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4"


def _start_engine():
    """Spawn the engine subprocess."""
    return subprocess.Popen(  # noqa: S603
        _ENGINE_CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1,  # Line buffered
    )


def _send(process, cmd):
    """Send a command to the engine."""
    print(f"> {cmd}")
    process.stdin.write(cmd + "\n")
    process.stdin.flush()


def _expect(process, pattern, timeout=2.0):
    """Wait for *pattern* to appear in engine output."""
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
    """Execute the sequence of UCI checks; return *True* on success."""
    # verify 'uci'
    _send(process, "uci")
    if not _expect(process, "id name ChessEngine"):
        return False
    if not _expect(process, "uciok"):
        return False

    # verify 'isready'
    _send(process, "isready")
    if not _expect(process, "readyok"):
        return False

    # verify 'position startpos' and 'go depth 1'
    _send(process, "position startpos")
    _send(process, "go depth 1")
    if not _expect(process, "bestmove"):
        return False

    # verify 'position fen ... moves ...'
    _send(process, f"position fen {_START_FEN}")
    _send(process, "go depth 1")
    return _expect(process, "bestmove")


def verify_uci():
    """Run the full UCI verification sequence."""
    process = _start_engine()
    ok = _run_checks(process)

    if not ok:
        process.terminate()
        return False

    # verify 'quit'
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
