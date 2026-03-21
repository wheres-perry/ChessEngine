from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from elo_tests.config import RunConfig
from elo_tests.engines.factory import create_engine
from elo_tests.game import game_record_to_json, play_game
from elo_tests.models import EloSummary
from elo_tests.openings import load_opening_fens
from elo_tests.reports import JSONLReporter, write_summary_json, write_summary_markdown
from elo_tests.scheduler import build_paired_schedule
from elo_tests.stats.bootstrap import paired_bootstrap_elo_ci
from elo_tests.stats.elo import mean_score, normal_ci_for_elo, score_rate_to_elo
from elo_tests.stats.sequential import should_stop


def _ci_halfwidth(ci: tuple[float, float]) -> float:
    return (ci[1] - ci[0]) / 2.0


def run_elo_estimation(config: RunConfig) -> EloSummary:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_{config.seed}"
    out_dir = Path(config.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "games.jsonl"
    reporter = JSONLReporter(results_path)

    openings = load_opening_fens(config.openings_file)
    schedule = build_paired_schedule(
        openings=openings,
        paired_blocks_target=config.paired_blocks_target,
        seed=config.seed,
    )

    candidate = create_engine(config.candidate, run_seed=config.seed)
    baseline = create_engine(config.baseline, run_seed=config.seed + 1)

    pair_scores_by_id: dict[int, list[float]] = defaultdict(list)
    game_scores: list[float] = []

    primary_ci = (0.0, 0.0)
    normal_ci = (0.0, 0.0)
    bootstrap_ci = (0.0, 0.0)
    stop_reason = "max_schedule_reached"
    stopped_early = False

    for game in schedule:
        if len(game_scores) >= config.max_games:
            stop_reason = "max_games_reached"
            break

        record = play_game(
            run_id=run_id,
            candidate=candidate,
            baseline=baseline,
            scheduled=game,
            tc=config.time_control,
        )
        reporter.append(game_record_to_json(record))

        game_scores.append(record.candidate_score)
        pair_scores_by_id[record.pair_id].append(record.candidate_score)

        pairs_played = len(pair_scores_by_id)
        games_played = len(game_scores)
        should_recompute = (
            pairs_played > 0 and pairs_played % config.recompute_every_pairs == 0
        )
        if not should_recompute:
            continue

        pair_means = [sum(pair_scores_by_id[idx]) / len(pair_scores_by_id[idx]) for idx in sorted(pair_scores_by_id)]
        normal_ci = normal_ci_for_elo(game_scores, level=config.ci_level)
        bootstrap_ci = paired_bootstrap_elo_ci(
            pair_scores=pair_means,
            level=config.ci_level,
            n_resamples=config.bootstrap_resamples,
            seed=config.seed + pairs_played,
        )
        primary_ci = bootstrap_ci if config.ci_method == "bootstrap" else normal_ci

        stop, reason = should_stop(
            pairs_played=pairs_played,
            games_played=games_played,
            min_blocks=config.min_blocks,
            max_games=config.max_games,
            ci_halfwidth_elo=_ci_halfwidth(primary_ci),
            target_halfwidth_elo=config.stop_ci_halfwidth_elo,
        )
        if stop:
            stop_reason = reason
            stopped_early = reason != "max_games_reached"
            break

    if not game_scores:
        game_scores = [0.5]
    if primary_ci == (0.0, 0.0):
        normal_ci = normal_ci_for_elo(game_scores, level=config.ci_level)
        pair_means = [sum(pair_scores_by_id[idx]) / len(pair_scores_by_id[idx]) for idx in sorted(pair_scores_by_id)] or [0.5]
        bootstrap_ci = paired_bootstrap_elo_ci(
            pair_scores=pair_means,
            level=config.ci_level,
            n_resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        primary_ci = bootstrap_ci if config.ci_method == "bootstrap" else normal_ci

    score = mean_score(game_scores)
    summary = EloSummary(
        run_id=run_id,
        seed=config.seed,
        candidate_id=candidate.engine_id,
        baseline_id=baseline.engine_id,
        games_played=len(game_scores),
        pairs_played=len(pair_scores_by_id),
        mean_score=score,
        elo_point=score_rate_to_elo(score),
        ci_level=config.ci_level,
        ci_primary_low=primary_ci[0],
        ci_primary_high=primary_ci[1],
        ci_normal_low=normal_ci[0],
        ci_normal_high=normal_ci[1],
        ci_bootstrap_low=bootstrap_ci[0],
        ci_bootstrap_high=bootstrap_ci[1],
        ci_method=config.ci_method,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        results_path=str(results_path),
    )

    write_summary_json(summary, out_dir / "summary.json")
    write_summary_markdown(summary, out_dir / "summary.md")
    return summary
