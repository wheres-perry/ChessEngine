"""UCI Protocol Handler."""

import logging
import sys
from collections.abc import Callable
from typing import ClassVar

from engine.config import EngineConfig
from engine.factory import STARTING_FEN, create_engine_runtime

_MAX_DEPTH = 100


def _out(msg: str) -> None:
    """Output to stdout with flush."""
    print(msg, flush=True)


class UCIHandler:
    """Handles UCI commands and interacts with the engine."""

    def __init__(self) -> None:
        self.config = EngineConfig()
        self.runtime = create_engine_runtime(self.config)

    def _dispatch(self, line: str) -> None:
        """Route a single UCI command line to the appropriate handler."""
        parts = line.strip().split()
        if not parts:
            return

        command = parts[0]
        args = parts[1:]

        if command == "quit":
            sys.exit(0)

        handler = self._commands.get(command)
        if handler is not None:
            handler(self, args)

    def _ponderhit(self, _args: list[str] | None = None) -> None:
        """Handle ponderhit (no-op)."""

    def _uci(self, _args: list[str] | None = None) -> None:
        """Handle uci identification handshake."""
        _out("id name ChessEngine")
        _out("id author Copilot")
        _out("option name Hash type spin default 16 min 1 max 1024")
        _out("uciok")

    def _isready(self, _args: list[str] | None = None) -> None:
        """Handle isready."""
        _out("readyok")

    def _ucinewgame(self, _args: list[str] | None = None) -> None:
        """Handle ucinewgame."""
        self.runtime.searcher.reset()

    def _position(self, args: list[str] | None = None) -> None:
        """Handle position command."""
        if not args:
            return

        fen_start = STARTING_FEN
        moves_idx = -1

        if args[0] == "startpos":
            fen_start = STARTING_FEN
            if len(args) > 1 and args[1] == "moves":
                moves_idx = 2
        elif args[0] == "fen":
            if "moves" in args:
                try:
                    kw_idx = args.index("moves")
                    fen_parts = args[1:kw_idx]
                    fen_start = " ".join(fen_parts)
                    moves_idx = kw_idx + 1
                except ValueError:
                    pass
            else:
                fen_parts = args[1:]
                fen_start = " ".join(fen_parts)

        self.runtime.board.from_fen(fen_start)

        if moves_idx != -1 and moves_idx < len(args):
            for move in args[moves_idx:]:
                self.runtime.board.push_uci(move)

    def _go(self, args: list[str] | None = None) -> None:
        """Handle go command."""
        if args is None:
            args = []
        depth = 4
        if "depth" in args:
            try:
                idx = args.index("depth")
                depth = int(args[idx + 1])
            except (ValueError, IndexError):
                pass

        _score, best_move = self.runtime.searcher.search(depth)

        if best_move:
            _out(f"bestmove {best_move}")
        else:
            _out("bestmove 0000")

    def _stop(self, _args: list[str] | None = None) -> None:
        """Handle stop (no-op)."""

    def _setoption(self, args: list[str] | None = None) -> None:
        """Handle setoption command."""
        if not args:
            return
        try:
            name_idx = args.index("name")
        except ValueError:
            return

        if "value" in args:
            value_idx = args.index("value")
            name_parts = args[name_idx + 1 : value_idx]
            value_parts = args[value_idx + 1 :]
            value = " ".join(value_parts)
        else:
            name_parts = args[name_idx + 1 :]
            value = "true"

        name = " ".join(name_parts)

        if name == "Hash":
            try:
                # Update config.search.tt_size_mb if it exists
                if hasattr(self.config.search, "tt_size_mb"):
                    self.config.search.tt_size_mb = int(value)
            except ValueError:
                pass

    _CommandHandler = Callable[["UCIHandler", list[str] | None], None]

    _commands: ClassVar[dict[str, _CommandHandler]] = {
        "uci": _uci,
        "isready": _isready,
        "ucinewgame": _ucinewgame,
        "position": _position,
        "go": _go,
        "stop": _stop,
        "setoption": _setoption,
        "ponderhit": _ponderhit,
    }


def main() -> None:
    handler = UCIHandler()
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            handler._dispatch(line)
        except KeyboardInterrupt:
            break
        except (ValueError, OSError):
            logging.exception("Error processing UCI command")
            continue


if __name__ == "__main__":
    main()
