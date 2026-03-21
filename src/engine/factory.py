"""Factory functions and adapter classes for assembling engine runtime objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from engine._core import chess_engine_core as core
from engine.config_solver import ConfigSolver
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

    def from_fen(self, fen: str) -> Any:
        """Load position from FEN string."""
        ...

    def fen(self) -> str:
        """Return current position as FEN string."""
        ...

    def push_uci(self, uci: str) -> None:
        """Make a move in UCI notation."""
        ...

    def legal_moves(self) -> list[str]:
        """Return list of legal moves in UCI notation."""
        ...

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        ...

    def get_internal_board(self) -> Any:
        """Return the internal board object."""
        ...


class CoreBoardAdapter(CoreAdapter):
    """Adapter for the C++ Core Board."""

    def __init__(self, board: core.Board):
        self.board = board

    def from_fen(self, fen: str) -> Any:
        """Load position from FEN string."""
        self.board.set_fen(fen)
        return self

    def fen(self) -> str:
        """Return current position as FEN string."""
        return self.board.fen()

    def push_uci(self, uci: str) -> None:
        """Make a move in UCI notation."""
        self.board.push(core.Move.from_uci(uci))

    def legal_moves(self) -> list[str]:
        """Return list of legal moves in UCI notation."""
        return [m.uci() for m in self.board.generate_legal_moves()]

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return bool(self.board.is_game_over() != core.GameState.ONGOING)

    def get_internal_board(self) -> core.Board:
        """Return the internal board object."""
        return self.board


@runtime_checkable
class SearchAdapter(Protocol):
    """Protocol for the search engine."""

    def search(self, depth: int) -> tuple[float | None, str | None]:
        """Search for the best move at the given depth."""
        ...

    def get_stats(self) -> SearchStats:
        """Return search statistics."""
        ...

    def reset(self) -> None:
        """Reset the search state."""
        ...


class PythonSearchAdapter(SearchAdapter):
    """Adapter for the Python Minimax searcher."""

    def __init__(self, engine: Minimax):
        self.engine = engine

    def search(self, depth: int) -> tuple[float | None, str | None]:
        """Search for the best move at the given depth."""
        score, move = self.engine.find_best_move(depth)
        return score, move.uci() if move else None

    def get_stats(self) -> SearchStats:
        """Return search statistics."""
        return self.engine.stats

    def reset(self) -> None:
        """Reset the search state."""
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
        """Search for the best move and return score and UCI move.

        Args:
            depth: Search depth (uses config default if None).

        Returns:
            Tuple of (score, uci_move) or (None, None) if no move found.
        """
        score, move = self.find_best_move(depth)
        return score, move.uci() if move else None

    @property
    def stats(self) -> SearchStats:
        """Return search statistics."""
        return self.searcher.stats

    def reset(self) -> None:
        """Reset search state, clearing history and TT."""
        self.searcher.reset_state()


def create_core_adapter(config: EngineConfig, fen: str | None = None) -> CoreAdapter:
    """Create a CoreBoardAdapter from the given FEN (or the starting position).

    Args:
        config: Engine configuration (currently unused but reserved for future use).
        fen: FEN string to load, or None to use the starting position.

    Returns:
        A CoreAdapter wrapping the initialized board.
    """
    _ = config
    start_fen = fen if fen else STARTING_FEN

    board = core.Board.from_fen(fen) if fen else core.Board.from_fen(start_fen)
    return CoreBoardAdapter(board)


def create_engine(config: EngineConfig, fen: str | None = None) -> Engine:
    """Construct the primary `Engine` object.

    Args:
        config: Engine configuration to use.
        fen: FEN string to load, or None to use the starting position.

    Returns:
        An initialized Engine instance ready for search operations.
    """
    ConfigSolver(config).solve()

    board = core.Board.from_fen(fen) if fen else core.Board.from_fen(STARTING_FEN)
    evaluator = EvaluatorFactory.create(config.evaluation)
    searcher = Minimax(board, evaluator, config)
    return Engine(board=board, evaluator=evaluator, searcher=searcher, config=config)


def create_search_adapter(
    config: EngineConfig, core_adapter: CoreAdapter
) -> SearchAdapter:
    """Build a Python search adapter from the given config and board adapter.

    Args:
        config: Engine configuration to use.
        core_adapter: Board adapter providing access to the internal board.

    Returns:
        A SearchAdapter wrapping the configured search engine.

    Raises:
        ValueError: If the core_adapter is not a CoreBoardAdapter.
    """
    ConfigSolver(config).solve()

    if not isinstance(core_adapter, CoreBoardAdapter):
        raise ValueError("Search requires Core backend board.")

    board = core_adapter.get_internal_board()
    evaluator = EvaluatorFactory.create(config.evaluation)
    engine = Minimax(board, evaluator, config)
    return PythonSearchAdapter(engine)


def create_engine_runtime(
    config: EngineConfig, fen: str | None = None
) -> EngineRuntime:
    """Canonical entry point to build the engine.

    Args:
        config: Engine configuration to use.
        fen: FEN string to load, or None to use the starting position.

    Returns:
        An EngineRuntime containing all initialized components.
    """
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
