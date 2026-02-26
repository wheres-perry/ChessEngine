"""Unit tests for the UCI protocol handler (engine.uci.UCIHandler)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from engine.uci import _MAX_DEPTH, UCIHandler

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
# uci command
# ---------------------------------------------------------------------------


def test_uci_sends_id_and_uciok() -> None:
    """'uci' must emit id name, id author, at least one option, and 'uciok'."""
    output = _dispatch_and_capture(_handler(), "uci")
    assert any(line.startswith("id name ") for line in output)
    assert any(line.startswith("id author ") for line in output)
    assert any(line.startswith("option ") for line in output)
    assert output[-1] == "uciok"


def test_uci_exposes_hash_option() -> None:
    """Verify the Hash option is advertised."""
    output = _dispatch_and_capture(_handler(), "uci")
    option_lines = [ln for ln in output if ln.startswith("option")]
    names = [ln.split("name ")[1].split(" type")[0] for ln in option_lines]
    assert "Hash" in names


# ---------------------------------------------------------------------------
# isready command
# ---------------------------------------------------------------------------


def test_isready_sends_readyok() -> None:
    """'isready' must emit 'readyok'."""
    output = _dispatch_and_capture(_handler(), "isready")
    assert output == ["readyok"]


# ---------------------------------------------------------------------------
# position command
# ---------------------------------------------------------------------------


def test_position_startpos() -> None:
    """'position startpos' sets board to initial state."""
    h = _handler()
    # Mock the runtime board
    h.runtime.board = MagicMock()

    _dispatch_and_capture(h, "position startpos")

    # Verify from_fen was called with startpos
    # (which is internal detail but we can check args).
    # The default STARTING_FEN is usually what's passed.
    h.runtime.board.from_fen.assert_called()


def test_position_fen() -> None:
    """'position fen ...' sets board to specific FEN."""
    h = _handler()
    h.runtime.board = MagicMock()

    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    _dispatch_and_capture(h, f"position fen {fen}")

    h.runtime.board.from_fen.assert_called_with(fen)


def test_position_moves() -> None:
    """'position ... moves ...' applies moves."""
    h = _handler()
    h.runtime.board = MagicMock()

    _dispatch_and_capture(h, "position startpos moves e2e4 e7e5")

    # from_fen called once
    h.runtime.board.from_fen.assert_called_once()
    # push_uci called twice
    assert h.runtime.board.push_uci.call_count == 2
    h.runtime.board.push_uci.assert_any_call("e2e4")
    h.runtime.board.push_uci.assert_any_call("e7e5")


# ---------------------------------------------------------------------------
# go command
# ---------------------------------------------------------------------------


def test_go_depth() -> None:
    """'go depth X' calls search with depth X."""
    h = _handler()
    h.runtime.searcher = MagicMock()
    h.runtime.searcher.search.return_value = (0.0, "e2e4")

    _dispatch_and_capture(h, "go depth 5")

    h.runtime.searcher.search.assert_called_with(5)


def test_go_infinite_is_not_supported_yet_but_uses_default() -> None:
    """'go' without depth uses default (currently 4)."""
    h = _handler()
    h.runtime.searcher = MagicMock()
    h.runtime.searcher.search.return_value = (None, None)

    _dispatch_and_capture(h, "go")

    # Default depth is 4 in our implementation
    h.runtime.searcher.search.assert_called_with(4)


# ---------------------------------------------------------------------------
# setoption command
# ---------------------------------------------------------------------------


def test_setoption_hash() -> None:
    """Test setting Hash option updates config."""
    h = _handler()
    # We need to ensure h.runtime.config.search exists and has tt_size_mb
    # Assuming the real classes structure:
    # h.config.search.tt_size_mb

    # Let's mock the config object to just check if attribute is set
    h.config.search = MagicMock()

    _dispatch_and_capture(h, "setoption name Hash value 64")

    assert h.config.search.tt_size_mb == 64
