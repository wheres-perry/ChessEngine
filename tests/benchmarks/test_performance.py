"""
Performance benchmarks for the C++ core engine.
Tests include heavy stress tests like Perft, deep game simulation,
search node expansion, and raw move throughput.

Designed for direct API usage with no abstraction overhead.
Refactored to use the Factory pattern for board creation.
"""

from __future__ import annotations

import random
from typing import Any, cast

import pytest

from engine import EngineConfig, create_engine_runtime
from engine._core import chess_engine_core as core
from engine.factory import CoreAdapter, CoreBoardAdapter, create_core_adapter

# ============================================================================
# CONSTANTS & DATA
# ============================================================================

# A mix of opening, middlegame, and endgame positions for varied testing
STRESS_FENS = [
    # Start
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # KiwiPete (Perft benchmark standard) - High branching factor
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    # Position 5 (Endgame / edge cases)
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    # Complex Middlegame
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
]

# Fixed seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def deep_game_moves() -> list[str]:
    """
    Pre-generates a valid sequence of 300 moves (150 full moves) from start pos.
    Uses core to ensure validity, returns UCI strings.
    """
    board = core.Board()
    rng = random.Random(RANDOM_SEED)
    moves: list[str] = []

    # Generate up to 300 ply
    for _ in range(300):
        if board.is_game_over() != core.GameState.ONGOING:
            break
        legal = board.generate_legal_moves()
        if not legal:
            break
        move = rng.choice(legal)
        moves.append(core.move_to_uci(move))
        board.push(move)

    return moves


@pytest.fixture
def board_adapter_factory():
    """Fixture to provide a factory function for board adapters."""
    config = EngineConfig()

    def _create(fen: str | None = None) -> CoreAdapter:
        return create_core_adapter(config, fen)

    return _create


# ============================================================================
# BENCHMARK TESTS
# ============================================================================


def test_full_game_cycle_300_ply(benchmark: Any, deep_game_moves: list[str]) -> None:
    """
    STRESS TEST: Push a full 300-ply game and then Pop it all back.
    Tests history stack depth, repetition tracking, and accumulated state updates.
    """
    uci_moves = deep_game_moves
    config = EngineConfig()

    def run_cycle() -> None:
        # Create via Factory
        board = create_core_adapter(config)

        # Push 300 moves
        for uci in uci_moves:
            board.push_uci(uci)

        # Pop 300 moves - Adapter doesn't expose pop() yet?
        # Let's check CoreAdapter protocol.
        # It needs pop() for this test.
        # The underlying boards support it.
        # We should probably add pop() to the adapter if we want to be strict.
        # For now, accessing internal board.
        internal_board = board.get_internal_board()
        for _ in range(len(uci_moves)):
            internal_board.pop()

    benchmark(run_cycle)


def test_perft_traversal_depth_3(benchmark: Any) -> None:
    """
    STRESS TEST: Recursive Perft to Depth 3 on 'KiwiPete'.
    """
    fen = STRESS_FENS[1]
    depth = 3
    config = EngineConfig()

    # We need a recursive function.
    # Using the adapter is slightly slower due to dynamic dispatch,
    # but that's what we want to measure (the full stack cost).
    # However, Perft is usually internal.
    # For fair comparison with previous native tests, we should probably
    # measure the native perft if available, or the recursive movegen.

    # If we use internal board, we match previous behavior.

    board_adapter = create_core_adapter(config, fen)
    root_board = board_adapter.get_internal_board()

    def perft_core(board, d: int) -> int:
        if d == 0:
            return 1
        nodes = 0
        moves = board.generate_legal_moves()
        for move in moves:
            board.push(move)
            nodes += perft_core(board, d - 1)
            board.pop()
        return nodes

    def run_benchmark() -> None:
        # Re-create or clone to ensure clean state if needed?
        # Perft restores state, so one instance is fine.
        perft_core(root_board, depth)

    benchmark.pedantic(run_benchmark, rounds=5, iterations=1)


def test_perft_traversal_depth_5(benchmark: Any) -> None:
    """
    STRESS TEST: Recursive Perft to Depth 5 on start position.
    """
    fen = STRESS_FENS[0]
    depth = 5
    config = EngineConfig()

    board_adapter = create_core_adapter(config, fen)
    root_board = board_adapter.get_internal_board()

    def perft_core(board, d: int) -> int:
        if d == 0:
            return 1
        nodes = 0
        moves = board.generate_legal_moves()
        for move in moves:
            board.push(move)
            nodes += perft_core(board, d - 1)
            board.pop()
        return nodes

    def run_benchmark() -> None:
        perft_core(root_board, depth)

    benchmark.pedantic(run_benchmark, rounds=3, iterations=1)


def test_search_node_expansion_loop(benchmark: Any) -> None:
    """
    STRESS TEST: Simulate Alpha-Beta Inner Loop.
    """
    fen = STRESS_FENS[1]
    loops = 200
    config = EngineConfig()

    board_adapter = create_core_adapter(config, fen)
    board = board_adapter.get_internal_board()

    def run_search_node() -> None:
        for _ in range(loops):
            moves = board.generate_legal_moves()
            for move in moves:
                board.push(move)
                _ = board.get_side_to_move()
                board.pop()

    benchmark(run_search_node)


def test_attacked_squares_middlegame(benchmark: Any) -> None:
    """
    MICROBENCH: Attacked-squares computation.
    """
    pytest.skip("core bindings do not expose get_attacked_squares yet")


def test_promotion_heavy_movegen(benchmark: Any) -> None:
    """
    MICROBENCH: Promotion-heavy move generation.
    """
    fen = "4k3/P1P1P1P1/8/8/8/8/p1p1p1p1/4K3 w - - 0 1"
    config = EngineConfig()

    board = create_core_adapter(config, fen)

    def run_gen() -> None:
        _ = board.legal_moves()

    benchmark.pedantic(run_gen, rounds=10, iterations=50)


def test_castling_and_ep_movegen(benchmark: Any) -> None:
    """
    MICROBENCH: Position with both castling rights and en-passant available.
    """
    fen = "r3k2r/ppp1pppp/8/3pP3/8/8/PPP1PPPP/R3K2R w KQkq d6 0 3"
    config = EngineConfig()

    board = create_core_adapter(config, fen)

    def run_gen() -> None:
        _ = board.legal_moves()

    benchmark.pedantic(run_gen, rounds=10, iterations=50)


def test_bulk_push_pop_precomputed(benchmark: Any) -> None:
    """
    MICROBENCH: Push/Pop a fixed precomputed move list repeatedly (no regen).
    """
    fen = STRESS_FENS[0]
    config = EngineConfig()

    board_adapter = create_core_adapter(config, fen)
    board = board_adapter.get_internal_board()

    moves = board.generate_legal_moves()

    def run_pp() -> None:
        for mv in moves:
            board.push(mv)
            board.pop()

    benchmark.pedantic(run_pp, rounds=10, iterations=20)


def test_bulk_instantiation_1000(benchmark: Any) -> None:
    """
    STRESS TEST: Create 1,000 distinct Board objects via Factory.
    """
    count = 1000
    fen = STRESS_FENS[1]
    config = EngineConfig()

    def run_alloc() -> None:
        for _ in range(count):
            _ = create_core_adapter(config, fen)

    benchmark(run_alloc)


def test_single_move_toggle_50k(benchmark: Any) -> None:
    """
    STRESS TEST: Push/Pop a single move 50,000 times.
    """
    iterations = 50_000
    config = EngineConfig()

    board_adapter = create_core_adapter(config)
    board = board_adapter.get_internal_board()

    move = core.Move(12, 28, 0)  # e2e4

    def run_toggle() -> None:
        for _ in range(iterations):
            board.push(move)
            board.pop()

    benchmark(run_toggle)


def test_copy_chain_stress(benchmark: Any) -> None:
    """
    STRESS TEST: Chain copies of the board.
    """
    depth = 1000
    fen = STRESS_FENS[3]
    config = EngineConfig()

    board_adapter = create_core_adapter(config, fen)
    root = board_adapter.get_internal_board()

    def run_chain() -> None:
        b = root
        for _ in range(depth):
            b = b.copy()

    benchmark(run_chain)


def test_batch_legal_generation(benchmark: Any) -> None:
    """
    STRESS TEST: Pure move generation throughput via Adapter.
    """
    loops = 100
    config = EngineConfig()

    # Create adapters
    adapters = [create_core_adapter(config, f) for f in STRESS_FENS]

    def run_gen() -> None:
        for _ in range(loops):
            for b in adapters:
                _ = b.legal_moves()

    benchmark(run_gen)
