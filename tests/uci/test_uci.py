"""Unit tests for the UCI protocol handler (engine.uci.UCIHandler)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from engine.uci import _MAX_DEPTH, UCIHandler, main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _handler() -> UCIHandler:
    """Return a fresh UCIHandler with no engine instantiated yet."""
    return UCIHandler()


def _dispatch_and_capture(handler: UCIHandler, *lines: str) -> list[str]:
    """Send one or more UCI lines and return all emitted output lines."""
    captured: list[str] = []

    # We mock _out to capture output
    with patch("engine.uci._out", side_effect=lambda msg: captured.append(msg)):
        for line in lines:
            handler._dispatch(line)
    return captured


# ---------------------------------------------------------------------------
# Core dispatch and loop
# ---------------------------------------------------------------------------

def test_dispatch_empty_line() -> None:
    """Empty line should be ignored."""
    h = _handler()
    h._dispatch("")
    h._dispatch("   ")

def test_dispatch_quit() -> None:
    """quit command should call sys.exit."""
    h = _handler()
    with patch("sys.exit") as mock_exit:
        h._dispatch("quit")
        mock_exit.assert_called_once_with(0)

def test_main_loop() -> None:
    """Test the main loop handling normal input and EOF."""
    # Mock sys.stdin.readline to return a few commands then empty string (EOF)
    inputs = ["isready\n", "uci\n", ""]
    with patch("sys.stdin.readline", side_effect=inputs):
        with patch("engine.uci._out") as mock_out:
            main()
            # isready and uci both output something
            assert mock_out.call_count >= 2

def test_main_loop_keyboard_interrupt() -> None:
    """Test main loop exits gracefully on KeyboardInterrupt."""
    with patch("sys.stdin.readline", side_effect=KeyboardInterrupt):
        main()  # Should not raise

def test_main_loop_value_error() -> None:
    """Test main loop catches ValueError and continues."""
    inputs = ["bad_command\n", ""]
    with patch("sys.stdin.readline", side_effect=inputs):
        with patch.object(UCIHandler, "_dispatch", side_effect=[ValueError("test"), None]):
            with patch("logging.exception") as mock_log:
                main()
                mock_log.assert_called_once()

# ---------------------------------------------------------------------------
# uci & isready
# ---------------------------------------------------------------------------

def test_uci_sends_id_and_uciok() -> None:
    """'uci' must emit id name, id author, at least one option, and 'uciok'."""
    output = _dispatch_and_capture(_handler(), "uci")
    assert any(line.startswith("id name ") for line in output)
    assert any(line.startswith("id author ") for line in output)
    assert any(line.startswith("option ") for line in output)
    assert output[-1] == "uciok"

def test_isready_sends_readyok() -> None:
    """'isready' must emit 'readyok'."""
    output = _dispatch_and_capture(_handler(), "isready")
    assert output == ["readyok"]

def test_ucinewgame() -> None:
    """'ucinewgame' resets the searcher."""
    h = _handler()
    h.runtime.searcher = MagicMock()
    h._dispatch("ucinewgame")
    h.runtime.searcher.reset.assert_called_once()

def test_ponderhit() -> None:
    """'ponderhit' is a no-op."""
    h = _handler()
    h._dispatch("ponderhit")

def test_stop() -> None:
    """'stop' is a no-op."""
    h = _handler()
    h._dispatch("stop")

# ---------------------------------------------------------------------------
# position command
# ---------------------------------------------------------------------------

def test_position_empty() -> None:
    """'position' with no args does nothing."""
    h = _handler()
    h.runtime.board = MagicMock()
    h._dispatch("position")
    h.runtime.board.from_fen.assert_not_called()

def test_position_startpos() -> None:
    """'position startpos' sets board to initial state."""
    h = _handler()
    h.runtime.board = MagicMock()
    _dispatch_and_capture(h, "position startpos")
    h.runtime.board.from_fen.assert_called()

def test_position_fen() -> None:
    """'position fen ...' sets board to specific FEN."""
    h = _handler()
    h.runtime.board = MagicMock()
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    _dispatch_and_capture(h, f"position fen {fen}")
    h.runtime.board.from_fen.assert_called_with(fen)

def test_position_fen_with_moves() -> None:
    """'position fen ... moves ...' sets board to FEN and applies moves."""
    h = _handler()
    h.runtime.board = MagicMock()
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    _dispatch_and_capture(h, f"position fen {fen} moves e7e5")
    h.runtime.board.from_fen.assert_called_with(fen)
    h.runtime.board.push_uci.assert_called_with("e7e5")

def test_position_fen_with_moves_bad_index() -> None:
    """'position fen moves' without actual fen string handles it gracefully."""
    h = _handler()
    h.runtime.board = MagicMock()
    # "moves" is index 1. fen_parts will be empty.
    _dispatch_and_capture(h, "position fen moves e2e4")
    # Should join empty string and call from_fen("")
    h.runtime.board.from_fen.assert_called_with("")
    h.runtime.board.push_uci.assert_called_with("e2e4")

def test_position_moves() -> None:
    """'position startpos moves ...' applies moves."""
    h = _handler()
    h.runtime.board = MagicMock()
    _dispatch_and_capture(h, "position startpos moves e2e4 e7e5")
    h.runtime.board.from_fen.assert_called_once()
    assert h.runtime.board.push_uci.call_count == 2
    h.runtime.board.push_uci.assert_any_call("e2e4")
    h.runtime.board.push_uci.assert_any_call("e7e5")

# ---------------------------------------------------------------------------
# go command
# ---------------------------------------------------------------------------

def test_go_no_args() -> None:
    """'go' without args uses default depth."""
    h = _handler()
    h.runtime.searcher = MagicMock()
    h.runtime.searcher.search.return_value = (0.0, "e2e4")
    _dispatch_and_capture(h, "go")
    h.runtime.searcher.search.assert_called_with(4)

def test_go_depth() -> None:
    """'go depth X' calls search with depth X."""
    h = _handler()
    h.runtime.searcher = MagicMock()
    h.runtime.searcher.search.return_value = (0.0, "e2e4")
    _dispatch_and_capture(h, "go depth 5")
    h.runtime.searcher.search.assert_called_with(5)

def test_go_depth_invalid() -> None:
    """'go depth invalid' falls back to default depth gracefully."""
    h = _handler()
    h.runtime.searcher = MagicMock()
    h.runtime.searcher.search.return_value = (0.0, "e2e4")
    _dispatch_and_capture(h, "go depth abc")
    h.runtime.searcher.search.assert_called_with(4)
    
def test_go_depth_missing_value() -> None:
    """'go depth' without a value falls back to default depth gracefully."""
    h = _handler()
    h.runtime.searcher = MagicMock()
    h.runtime.searcher.search.return_value = (0.0, "e2e4")
    _dispatch_and_capture(h, "go depth")
    h.runtime.searcher.search.assert_called_with(4)

def test_go_no_best_move() -> None:
    """'go' outputs bestmove 0000 if no move found."""
    h = _handler()
    h.runtime.searcher = MagicMock()
    h.runtime.searcher.search.return_value = (0.0, None)
    output = _dispatch_and_capture(h, "go depth 1")
    assert output == ["bestmove 0000"]

# ---------------------------------------------------------------------------
# setoption command
# ---------------------------------------------------------------------------

def test_setoption_no_args() -> None:
    """'setoption' with no args does nothing."""
    h = _handler()
    h._dispatch("setoption")

def test_setoption_no_name() -> None:
    """'setoption' without 'name' keyword does nothing."""
    h = _handler()
    h._dispatch("setoption value 64")

def test_setoption_hash() -> None:
    """Test setting Hash option updates config."""
    h = _handler()
    h.config.search = MagicMock()
    _dispatch_and_capture(h, "setoption name Hash value 64")
    assert h.config.search.tt_size_mb == 64

def test_setoption_hash_invalid_value() -> None:
    """Test setting Hash option to invalid value doesn't crash."""
    h = _handler()
    h.config.search = MagicMock()
    h.config.search.tt_size_mb = 16
    _dispatch_and_capture(h, "setoption name Hash value abc")
    assert h.config.search.tt_size_mb == 16  # Should not change

def test_setoption_hash_no_value() -> None:
    """Test setting Hash without 'value' keyword assumes true/default behavior."""
    h = _handler()
    h.config.search = MagicMock()
    # 'true' cannot be cast to int, so value shouldn't update tt_size_mb but shouldn't crash
    h.config.search.tt_size_mb = 16
    _dispatch_and_capture(h, "setoption name Hash")
    assert h.config.search.tt_size_mb == 16
