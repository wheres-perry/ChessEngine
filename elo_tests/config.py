from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from elo_tests.models import EngineSpec, RunConfig, TimeControlSpec


def default_run_config() -> RunConfig:
    return RunConfig(
        candidate=EngineSpec(engine_id="candidate", strength_elo=40.0, version="sim-1"),
        baseline=EngineSpec(engine_id="baseline", strength_elo=0.0, version="sim-1"),
        time_control=TimeControlSpec(movetime_ms=50, depth=None),
        openings_file=None,
        paired_blocks_target=200,
        min_blocks=100,
        max_games=800,
        ci_method="bootstrap",
        ci_level=0.95,
        stop_ci_halfwidth_elo=12.0,
        recompute_every_pairs=10,
        workers=1,
        seed=42,
        output_dir="elo_tests/output",
        bootstrap_resamples=2000,
    )


def _merge_engine(base: EngineSpec, payload: dict[str, Any]) -> EngineSpec:
    return EngineSpec(
        engine_id=str(payload.get("engine_id", base.engine_id)),
        kind="simulated",
        strength_elo=float(payload.get("strength_elo", base.strength_elo)),
        draw_bias=float(payload.get("draw_bias", base.draw_bias)),
        version=str(payload.get("version", base.version)),
    )


def load_run_config(config_path: str | None = None) -> RunConfig:
    base = default_run_config()
    if not config_path:
        return base

    path = Path(config_path)
    if not path.exists():
        return base

    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate_payload = payload.get("candidate", {})
    baseline_payload = payload.get("baseline", {})
    tc_payload = payload.get("time_control", {})

    return RunConfig(
        candidate=_merge_engine(base.candidate, candidate_payload),
        baseline=_merge_engine(base.baseline, baseline_payload),
        time_control=TimeControlSpec(
            movetime_ms=tc_payload.get("movetime_ms", base.time_control.movetime_ms),
            depth=tc_payload.get("depth", base.time_control.depth),
        ),
        openings_file=payload.get("openings_file", base.openings_file),
        paired_blocks_target=int(payload.get("paired_blocks_target", base.paired_blocks_target)),
        min_blocks=int(payload.get("min_blocks", base.min_blocks)),
        max_games=int(payload.get("max_games", base.max_games)),
        ci_method=payload.get("ci_method", base.ci_method),
        ci_level=float(payload.get("ci_level", base.ci_level)),
        stop_ci_halfwidth_elo=float(
            payload.get("stop_ci_halfwidth_elo", base.stop_ci_halfwidth_elo)
        ),
        recompute_every_pairs=int(
            payload.get("recompute_every_pairs", base.recompute_every_pairs)
        ),
        workers=int(payload.get("workers", base.workers)),
        seed=int(payload.get("seed", base.seed)),
        output_dir=str(payload.get("output_dir", base.output_dir)),
        bootstrap_resamples=int(
            payload.get("bootstrap_resamples", base.bootstrap_resamples)
        ),
    )
