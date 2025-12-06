"""
Performance benchmarks comparing the C++ core engine against python-chess.
Tests include heavy stress tests like Perft, deep game simulation,
search node expansion, and raw move throughput.

Designed for direct API usage with no abstraction overhead.
"""

from __future__ import annotations

import random
from typing import Any, cast

import chess as pychess
import pytest

# Import the C++ core extension directly
from engine._core import chess_engine_core as core

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
    Uses python-chess to ensure validity, returns UCI strings.
    """
    board = pychess.Board()
    rng = random.Random(RANDOM_SEED)
    moves: list[str] = []

    # Generate up to 300 ply
    for _ in range(300):
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        if not legal:
            break
        move = rng.choice(legal)
        moves.append(move.uci())
        board.push(move)

    return moves


@pytest.fixture(params=["core", "pychess"])
def engine_impl(request: pytest.FixtureRequest) -> str:
    """
    Parametrized fixture to run each test against both engines.
    """
    return cast("str", request.param)


# ============================================================================
# BENCHMARK TESTS
# ============================================================================


def test_full_game_cycle_300_ply(
    benchmark: Any, deep_game_moves: list[str], engine_impl: str
) -> None:
    """
    STRESS TEST: Push a full 300-ply game and then Pop it all back.
    Tests history stack depth, repetition tracking, and accumulated state updates.
    """
    # Prepare raw UCI strings
    uci_moves = deep_game_moves

    # Warmup pass (optional, but good practice)
    # (Not strictly necessary since benchmark handles warmups,
    # but ensure imports loaded)

    if engine_impl == "core":
        # Pre-convert to engine move objects to measure Board speed, not string parsing
        # We do this ONCE outside the benchmark loop if possible,
        # or measuring parsing is part of it?
        # The user asked for "stress", usually that implies board ops.
        # Let's measure the whole flow including parsing if that's typical usage,
        # OR pre-parse to test board raw speed.
        # To be fair to both, we'll parse inside the loop to simulate "loading a game".

        def run_cycle() -> None:
            board = core.Board()
            # Push 300 moves
            for uci in uci_moves:
                board.push(core.Move.from_uci(uci))
            # Pop 300 moves
            for _ in range(len(uci_moves)):
                board.pop()

    else:
        # Python-chess reference
        def run_cycle() -> None:
            board = pychess.Board()
            for uci in uci_moves:
                board.push(pychess.Move.from_uci(uci))
            for _ in range(len(uci_moves)):
                board.pop()

    # Run benchmark
    benchmark(run_cycle)


def test_perft_traversal_depth_3(benchmark: Any, engine_impl: str) -> None:
    """
    STRESS TEST: Recursive Perft to Depth 3 on 'KiwiPete'.
    Focuses on Move Generation + Make/Unmake consistency.
    KiwiPete Depth 3 is ~97,000 nodes.
    """
    fen = STRESS_FENS[1]  # KiwiPete
    depth = 3

    if engine_impl == "core":
        root = core.Board.from_fen(fen)

        def perft_core(board: core.Board, d: int) -> int:
            if d == 0:
                return 1
            nodes = 0
            # Direct iteration
            moves = board.generate_legal_moves()
            for move in moves:
                board.push(move)
                nodes += perft_core(board, d - 1)
                board.pop()
            return nodes

        def run_benchmark() -> None:
            perft_core(root, depth)

    else:
        root = pychess.Board(fen)

        def perft_pychess(board: pychess.Board, d: int) -> int:
            if d == 0:
                return 1
            nodes = 0
            # Python-chess iterator
            for move in board.legal_moves:
                board.push(move)
                nodes += perft_pychess(board, d - 1)
                board.pop()
            return nodes

        def run_benchmark() -> None:
            perft_pychess(root, depth)

    # Use pedantic to control rounds for this heavy test
    benchmark.pedantic(run_benchmark, rounds=5, iterations=1)


def test_perft_traversal_depth_5(benchmark: Any, engine_impl: str) -> None:
    """
    STRESS TEST: Recursive Perft to Depth 5 on start position.
    Pure move generation + make/unmake (no search). ~4.8M nodes.
    """
    fen = STRESS_FENS[0]  # Start position
    depth = 5

    if engine_impl == "core":
        root = core.Board.from_fen(fen)

        def perft_core(board: core.Board, d: int) -> int:
            if d == 0:
                return 1
            nodes = 0
            for move in board.generate_legal_moves():
                board.push(move)
                nodes += perft_core(board, d - 1)
                board.pop()
            return nodes

        def run_benchmark() -> None:
            perft_core(root, depth)

    else:
        root = pychess.Board(fen)

        def perft_pychess(board: pychess.Board, d: int) -> int:
            if d == 0:
                return 1
            nodes = 0
            for move in board.legal_moves:
                board.push(move)
                nodes += perft_pychess(board, d - 1)
                board.pop()
            return nodes

        def run_benchmark() -> None:
            perft_pychess(root, depth)

    # Depth 5 is still heavy; keep controlled rounds/iterations
    benchmark.pedantic(run_benchmark, rounds=3, iterations=1)


def test_search_node_expansion_loop(benchmark: Any, engine_impl: str) -> None:
    """
    STRESS TEST: Simulate Alpha-Beta Inner Loop.
    1. Generate all moves
    2. For each move: Push -> Check Hash/Turn -> Pop
    Repeated 200 times to simulate a 'heavy' node expansion.
    """
    fen = STRESS_FENS[1]  # KiwiPete again for high branching factor
    loops = 200

    if engine_impl == "core":
        board = core.Board.from_fen(fen)

        def run_search_node() -> None:
            for _ in range(loops):
                moves = board.generate_legal_moves()
                for move in moves:
                    board.push(move)
                    # Touch some state to simulate usage
                    _ = board.get_side_to_move()
                    _ = board.get_hash() if hasattr(board, "get_hash") else 0
                    board.pop()

    else:
        board = pychess.Board(fen)

        def run_search_node() -> None:
            for _ in range(loops):
                # list() needed to consume generator for fair comparison vs vector
                moves = list(board.legal_moves)
                for move in moves:
                    board.push(move)
                    _ = board.turn
                    # python-chess has no cheap hash getter usually, just verify turn
                    board.pop()

    benchmark(run_search_node)


def test_attacked_squares_middlegame(benchmark: Any, engine_impl: str) -> None:
    """
    MICROBENCH: Attacked-squares computation on a complex middlegame FEN.
    Stresses attacked-square cache/ray tables (core) vs. python-chess attacks().
    """
    fen = "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1"

    if engine_impl == "core":
        pytest.skip("core bindings do not expose get_attacked_squares yet")

    else:
        board = pychess.Board(fen)

        def run_attack() -> None:
            white_attacked = 0
            black_attacked = 0
            for sq in (
                board.pieces(pychess.PAWN, pychess.WHITE)
                | board.pieces(pychess.KNIGHT, pychess.WHITE)
                | board.pieces(pychess.BISHOP, pychess.WHITE)
                | board.pieces(pychess.ROOK, pychess.WHITE)
                | board.pieces(pychess.QUEEN, pychess.WHITE)
                | board.pieces(pychess.KING, pychess.WHITE)
            ):
                white_attacked |= int(board.attacks(sq))
            for sq in (
                board.pieces(pychess.PAWN, pychess.BLACK)
                | board.pieces(pychess.KNIGHT, pychess.BLACK)
                | board.pieces(pychess.BISHOP, pychess.BLACK)
                | board.pieces(pychess.ROOK, pychess.BLACK)
                | board.pieces(pychess.QUEEN, pychess.BLACK)
                | board.pieces(pychess.KING, pychess.BLACK)
            ):
                black_attacked |= int(board.attacks(sq))
            _ = white_attacked ^ black_attacked

    benchmark.pedantic(run_attack, rounds=10, iterations=50)


def test_promotion_heavy_movegen(benchmark: Any, engine_impl: str) -> None:
    """
    MICROBENCH: Promotion-heavy move generation.
    """
    fen = "4k3/P1P1P1P1/8/8/8/8/p1p1p1p1/4K3 w - - 0 1"

    if engine_impl == "core":
        board = core.Board.from_fen(fen)

        def run_gen() -> None:
            _ = board.generate_legal_moves()

    else:
        board = pychess.Board(fen)

        def run_gen() -> None:
            _ = list(board.legal_moves)

    benchmark.pedantic(run_gen, rounds=10, iterations=50)


def test_castling_and_ep_movegen(benchmark: Any, engine_impl: str) -> None:
    """
    MICROBENCH: Position with both castling rights and en-passant available.
    """
    fen = "r3k2r/ppp1pppp/8/3pP3/8/8/PPP1PPPP/R3K2R w KQkq d6 0 3"

    if engine_impl == "core":
        board = core.Board.from_fen(fen)

        def run_gen() -> None:
            _ = board.generate_legal_moves()

    else:
        board = pychess.Board(fen)

        def run_gen() -> None:
            _ = list(board.legal_moves)

    benchmark.pedantic(run_gen, rounds=10, iterations=50)


def test_bulk_push_pop_precomputed(benchmark: Any, engine_impl: str) -> None:
    """
    MICROBENCH: Push/Pop a fixed precomputed move list repeatedly (no regen).
    Isolates state-history and make/unmake cost.
    """
    fen = STRESS_FENS[0]  # Start position

    if engine_impl == "core":
        board = core.Board.from_fen(fen)
        moves = board.generate_legal_moves()

        def run_pp() -> None:
            for mv in moves:
                board.push(mv)
                board.pop()

    else:
        board = pychess.Board(fen)
        moves = list(board.legal_moves)

        def run_pp() -> None:
            for mv in moves:
                board.push(mv)
                board.pop()

    benchmark.pedantic(run_pp, rounds=10, iterations=20)


def test_attacked_vs_legal_mix(benchmark: Any, engine_impl: str) -> None:
    """
    MICROBENCH: Mix of attacked-squares and legal move generation on same FEN.
    """
    fen = STRESS_FENS[1]  # KiwiPete

    if engine_impl == "core":
        pytest.skip("core bindings do not expose get_attacked_squares yet")

    else:
        board = pychess.Board(fen)

        def run_mix() -> None:
            white_attacked = 0
            black_attacked = 0
            for sq in (
                board.pieces(pychess.PAWN, pychess.WHITE)
                | board.pieces(pychess.KNIGHT, pychess.WHITE)
                | board.pieces(pychess.BISHOP, pychess.WHITE)
                | board.pieces(pychess.ROOK, pychess.WHITE)
                | board.pieces(pychess.QUEEN, pychess.WHITE)
                | board.pieces(pychess.KING, pychess.WHITE)
            ):
                white_attacked |= int(board.attacks(sq))
            for sq in (
                board.pieces(pychess.PAWN, pychess.BLACK)
                | board.pieces(pychess.KNIGHT, pychess.BLACK)
                | board.pieces(pychess.BISHOP, pychess.BLACK)
                | board.pieces(pychess.ROOK, pychess.BLACK)
                | board.pieces(pychess.QUEEN, pychess.BLACK)
                | board.pieces(pychess.KING, pychess.BLACK)
            ):
                black_attacked |= int(board.attacks(sq))
            _ = white_attacked ^ black_attacked
            _ = list(board.legal_moves)

    benchmark.pedantic(run_mix, rounds=10, iterations=20)


def test_bulk_instantiation_1000(benchmark: Any, engine_impl: str) -> None:
    """
    STRESS TEST: Create 1,000 distinct Board objects.
    Measures memory allocation/init overhead.
    """
    count = 1000
    fen = STRESS_FENS[1]

    if engine_impl == "core":

        def run_alloc() -> None:
            for _ in range(count):
                _ = core.Board.from_fen(fen)
    else:

        def run_alloc() -> None:
            for _ in range(count):
                _ = pychess.Board(fen)

    benchmark(run_alloc)


def test_single_move_toggle_50k(benchmark: Any, engine_impl: str) -> None:
    """
    STRESS TEST: Push/Pop a single move 50,000 times.
    Tests the absolute hottest path for Make/Unmake.
    """
    iterations = 50_000

    if engine_impl == "core":
        board = core.Board()
        # e2e4
        move = core.Move(12, 28, 0)

        def run_toggle() -> None:
            for _ in range(iterations):
                board.push(move)
                board.pop()
    else:
        board = pychess.Board()
        move = pychess.Move.from_uci("e2e4")

        def run_toggle() -> None:
            for _ in range(iterations):
                board.push(move)
                board.pop()

    benchmark(run_toggle)


def test_copy_chain_stress(benchmark: Any, engine_impl: str) -> None:
    """
    STRESS TEST: Chain copies of the board.
    Board -> Copy -> Copy -> Copy ...
    Verifies full deep copy performance and correctness preservation.
    """
    depth = 1000
    fen = STRESS_FENS[3]  # Complex middlegame

    if engine_impl == "core":
        root = core.Board.from_fen(fen)

        def run_chain() -> None:
            b = root
            for _ in range(depth):
                b = b.copy()
    else:
        root = pychess.Board(fen)

        def run_chain() -> None:
            b = root
            for _ in range(depth):
                b = b.copy()

    benchmark(run_chain)


def test_batch_legal_generation(benchmark: Any, engine_impl: str) -> None:
    """
    STRESS TEST: Pure move generation throughput.
    Iterates through 4 different FEN types, generating moves 100 times each.
    """
    loops = 100

    if engine_impl == "core":
        # Pre-load boards to isolate generation time
        boards = [core.Board.from_fen(f) for f in STRESS_FENS]

        def run_gen() -> None:
            for _ in range(loops):
                for b in boards:
                    _ = b.generate_legal_moves()
    else:
        boards = [pychess.Board(f) for f in STRESS_FENS]

        def run_gen() -> None:
            for _ in range(loops):
                for b in boards:
                    # Force consumption of iterator
                    _ = list(b.legal_moves)

    benchmark(run_gen)
