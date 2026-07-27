"""Standalone Elo match runner for Moray configurations."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from engine._core import moray_core as chess
from engine.config import EngineConfig
from engine.factory import create_engine

OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqk2r/pppp1ppp/4pn2/8/2PP4/2P5/P3PPPP/R1BQKBNR w KQkq - 0 4",
    "r1bqk2r/pppp1ppp/2n2n2/4p3/2B1P3/2P2N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3",
]


@dataclass
class GameResult:
    """Outcome record of a single game."""

    opening_fen: str
    white_is_candidate: bool
    score: float  # 1.0 = Candidate Win, 0.5 = Draw, 0.0 = Baseline Win
    ply_count: int


def play_single_game(
    white_engine,
    black_engine,
    fen: str,
    max_plies: int = 150,
    depth: int = 4,
) -> tuple[float, int]:
    """Play a single game between white_engine and black_engine.

    Returns (white_score, total_plies).
    """
    white_engine.set_fen(fen)
    black_engine.set_fen(fen)
    board = chess.Board.from_fen(fen)

    for ply in range(max_plies):
        game_state = board.is_game_over()
        if game_state != chess.GameState.ONGOING:
            if game_state == chess.GameState.CHECKMATE:
                return (0.0 if board.get_side_to_move() else 1.0), ply
            return 0.5, ply

        current_engine = white_engine if board.get_side_to_move() else black_engine
        current_engine.board.set_fen(board.fen())

        _, move = current_engine.find_best_move(depth=depth)
        if not move:
            return 0.5, ply

        board.push(move)

    return 0.5, max_plies


def run_elo_match(
    num_pairs: int = 50,
    search_depth: int = 4,
) -> None:
    """Play a paired Elo match between Candidate and Baseline."""
    print(
        f"=== Starting Moray Elo Estimation Match "
        f"({num_pairs * 2} games, depth={search_depth}) ==="
    )
    start_time = time.time()

    candidate_config = EngineConfig()
    candidate_config.search_depth = search_depth

    baseline_config = EngineConfig()
    baseline_config.search_depth = search_depth
    baseline_config.evaluation.use_pst = False
    baseline_config.evaluation.use_pawn_structure = False
    baseline_config.evaluation.use_mobility = False
    baseline_config.evaluation.use_king_safety = False

    candidate_engine = create_engine(candidate_config)
    baseline_engine = create_engine(baseline_config)

    rng = random.Random(42)
    results: list[GameResult] = []

    wins, draws, losses = 0, 0, 0

    for i in range(num_pairs):
        fen = rng.choice(OPENINGS)

        # Game A: Candidate is White
        score_a, plies_a = play_single_game(
            candidate_engine, baseline_engine, fen, depth=search_depth
        )
        results.append(GameResult(fen, True, score_a, plies_a))  # noqa: FBT003
        if score_a == 1.0:
            wins += 1
        elif score_a == 0.5:
            draws += 1
        else:
            losses += 1

        # Game B: Candidate is Black
        white_score_b, plies_b = play_single_game(
            baseline_engine, candidate_engine, fen, depth=search_depth
        )
        score_b = 1.0 - white_score_b
        results.append(GameResult(fen, False, score_b, plies_b))  # noqa: FBT003
        if score_b == 1.0:
            wins += 1
        elif score_b == 0.5:
            draws += 1
        else:
            losses += 1

        if (i + 1) % 10 == 0 or i + 1 == num_pairs:
            played = len(results)
            curr_score = (wins + 0.5 * draws) / played
            print(
                f"Progress: {played}/{num_pairs * 2} games played | "
                f"W: {wins} D: {draws} L: {losses} | Score: {curr_score:.3f}"
            )

    elapsed = time.time() - start_time
    total_games = len(results)
    total_score = wins + 0.5 * draws
    score_rate = total_score / total_games

    if score_rate <= 0.0:
        elo_diff = -800.0
    elif score_rate >= 1.0:
        elo_diff = 800.0
    else:
        elo_diff = -400.0 * math.log10(1.0 / score_rate - 1.0)

    variance = max(1e-5, (score_rate * (1.0 - score_rate)) / total_games)
    se_score = math.sqrt(variance)
    margin_score = 1.96 * se_score

    score_low = max(0.001, score_rate - margin_score)
    score_high = min(0.999, score_rate + margin_score)

    elo_low = -400.0 * math.log10(1.0 / score_low - 1.0)
    elo_high = -400.0 * math.log10(1.0 / score_high - 1.0)

    print("\n================ FINAL ELO BENCHMARK REPORT ================")
    print(
        "Candidate Engine: Moray "
        "(Full Features: PST, Mobility, King Safety, Pawn Struct)"
    )
    print("Baseline Engine : Moray (Material-Only Baseline)")
    print(f"Match Format    : {total_games} Paired Games at Depth {search_depth}")
    print(
        f"Time Taken      : {elapsed:.2f} seconds "
        f"({total_games / elapsed:.1f} games/sec)"
    )
    print(f"Score Record    : +{wins} ={draws} -{losses} ({score_rate * 100:.1f}%)")
    print(f"Estimated Elo   : {elo_diff:+.1f} Elo")
    print(
        f"95% Conf. Int.  : [{elo_low:+.1f}, {elo_high:+.1f}] Elo "
        f"(±{(elo_high - elo_low) / 2:.1f})"
    )
    print("============================================================")


if __name__ == "__main__":
    run_elo_match(num_pairs=50, search_depth=4)
