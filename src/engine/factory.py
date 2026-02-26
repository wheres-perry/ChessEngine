"""Factory functions and adapter classes for assembling engine runtime objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from engine._core import chess_engine_core as core
from engine.config_dependency_resolver import DependencyResolver
from engine.evaluators import EvaluatorFactory
from engine.search.minimax import Minimax

if TYPE_CHECKING:
    from engine.config import EngineConfig
    from engine.evaluators import Evaluator
    from engine.search.stats import SearchStats

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@runtime_checkable
class CoreAdapter(Protocol):
    """Protocol for the chess board/core backend."""

    def from_fen(self, fen: str) -> Any: ...
    def fen(self) -> str: ...
    def push_uci(self, uci: str) -> None: ...
    def legal_moves(self) -> list[str]: ...
    def is_game_over(self) -> bool: ...
    def get_internal_board(self) -> Any: ...


class CoreBoardAdapter(CoreAdapter):
    """Adapter for the C++ Core Board."""

    def __init__(self, board: core.Board):
        self.board = board

    def from_fen(self, fen: str) -> Any:
        self.board.set_fen(fen)
        return self

    def fen(self) -> str:
        return self.board.fen()

    def push_uci(self, uci: str) -> None:
        self.board.push(core.Move.from_uci(uci))

    def legal_moves(self) -> list[str]:
        return [m.uci() for m in self.board.generate_legal_moves()]

    def is_game_over(self) -> bool:
        return bool(self.board.is_game_over() != core.GameState.ONGOING)

    def get_internal_board(self) -> core.Board:
        return self.board


@runtime_checkable
class SearchAdapter(Protocol):
    """Protocol for the search engine."""

    def search(self, depth: int) -> tuple[float | None, str | None]: ...
    def get_stats(self) -> SearchStats: ...
    def reset(self) -> None: ...


class PythonSearchAdapter(SearchAdapter):
    """Adapter for the Python Minimax searcher."""

    def __init__(self, engine: Minimax):
        self.engine = engine

    def search(self, depth: int) -> tuple[float | None, str | None]:
        score, move = self.engine.find_best_move(depth)
        return score, move.uci() if move else None

    def get_stats(self) -> SearchStats:
        # Minimax uses the python SearchStats directly
        return self.engine.stats

    def reset(self) -> None:
        self.engine.reset_state()


@dataclass
class EngineRuntime:
    """The runtime assembly of the engine."""

    board: CoreAdapter
    searcher: SearchAdapter
    evaluator: Evaluator | None  # Optional for C++ search if it has internal eval
    config: EngineConfig


class Engine:
    """Main Python engine orchestration layer.

    Keeps evaluator and search logic separate while exposing a single runtime object.
    """

    def __init__(
        self,
        board: core.Board,
        evaluator: Evaluator,
        searcher: Minimax,
        config: EngineConfig,
    ):
        self.board = board
        self.evaluator = evaluator
        self.searcher = searcher
        self.config = config

    def set_fen(self, fen: str) -> None:
        """Load a new position from a FEN string and rehash it."""
        self.board.set_fen(fen)
        if self.searcher.zobrist is not None:
            self.searcher.zobrist.hash_board(self.board)

    def push_uci(self, uci: str) -> None:
        """Apply a move in UCI notation and update the Zobrist hash."""
        self.board.push(core.Move.from_uci(uci))
        if self.searcher.zobrist is not None:
            self.searcher.zobrist.hash_board(self.board)

    def find_best_move(
        self,
        depth: int | None = None,
    ) -> tuple[float | None, core.Move | None]:
        """Search for the best move up to the given depth.

        Uses the config depth when depth is not specified.
        """
        search_depth = depth if depth is not None else self.config.search_depth
        return self.searcher.find_best_move(search_depth)

    def search(self, depth: int | None = None) -> tuple[float | None, str | None]:
        score, move = self.find_best_move(depth)
        return score, move.uci() if move else None

    @property
    def stats(self) -> SearchStats:
        return self.searcher.stats

    def reset(self) -> None:
        """Reset search state, clearing history and TT."""
        self.searcher.reset_state()


def create_core_adapter(config: EngineConfig, fen: str | None = None) -> CoreAdapter:
    """Create a CoreBoardAdapter from the given FEN (or the starting position)."""
    _ = config
    start_fen = fen if fen else STARTING_FEN

    # Create core board
    # Check if from_fen is available or init then load
    board = core.Board.from_fen(fen) if fen else core.Board.from_fen(start_fen)
    return CoreBoardAdapter(board)


def create_engine(config: EngineConfig, fen: str | None = None) -> Engine:
    """Construct the primary `Engine` object."""
    DependencyResolver(config).resolve()

    board = core.Board.from_fen(fen) if fen else core.Board.from_fen(STARTING_FEN)
    evaluator = EvaluatorFactory.create(config.evaluation)
    searcher = Minimax(board, evaluator, config)
    return Engine(board=board, evaluator=evaluator, searcher=searcher, config=config)


def create_search_adapter(
    config: EngineConfig, core_adapter: CoreAdapter
) -> SearchAdapter:
    """Build a Python search adapter from the given config and board adapter."""
    DependencyResolver(config).resolve()

    if not isinstance(core_adapter, CoreBoardAdapter):
        raise ValueError("Search requires Core backend board.")

    board = core_adapter.get_internal_board()
    evaluator = EvaluatorFactory.create(config.evaluation)
    engine = Minimax(board, evaluator, config)
    return PythonSearchAdapter(engine)


def create_engine_runtime(
    config: EngineConfig, fen: str | None = None
) -> EngineRuntime:
    """Canonical entry point to build the engine."""
    engine = create_engine(config, fen)

    board_adapter = CoreBoardAdapter(engine.board)
    search_adapter = PythonSearchAdapter(engine.searcher)

    # Extract evaluator if accessible (mainly for Python search)
    evaluator: Evaluator | None = None
    if isinstance(search_adapter, PythonSearchAdapter):
        evaluator = search_adapter.engine.evaluator

    return EngineRuntime(
        board=board_adapter, searcher=search_adapter, evaluator=evaluator, config=config
    )
