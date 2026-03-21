"""Infrastructure for benchmarking and baseline management."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SearchStats:
    """Metrics collected from a single search run."""

    nodes: int = 0
    depth: int = 0
    seldepth: int = 0
    effective_branching_factor: float = 0.0
    time_to_depth: float = 0.0
    nps: int = 0
    seldepth_ratio: float = 0.0

    beta_cutoffs: int = 0
    fmbc_count: int = 0
    fmbc_rate: float = 0.0
    killer_cuts: int = 0
    history_cuts: int = 0

    tt_hits: int = 0
    tt_hit_rate: float = 0.0
    hashfull_permillage: int = 0

    qsearch_nodes: int = 0
    qs_node_ratio: float = 0.0

    score: int = 0
    score_variance: float = 0.0
    best_move: str = ""

    time_stddev: float = 0.0
    nodes_stddev: float = 0.0

    pvs_researches: int = 0
    qs_see_pruning: int = 0
    root_move_changes: int = 0
    history_saturation: float = 0.0
    null_move_cuts: int = 0

    def calculate_derived_metrics(self):
        """Update rate metrics based on raw counts."""
        if self.beta_cutoffs > 0:
            self.fmbc_rate = self.fmbc_count / self.beta_cutoffs
        if self.nodes > 0:
            self.qs_node_ratio = self.qsearch_nodes / self.nodes
            self.tt_hit_rate = self.tt_hits / self.nodes

        if self.depth > 0 and self.nodes > 0:
            self.effective_branching_factor = self.nodes ** (1.0 / self.depth)

        if self.depth > 0:
            self.seldepth_ratio = self.seldepth / self.depth


@dataclass
class BenchmarkResult:
    """A full benchmark report for a specific configuration."""

    config_name: str
    timestamp: str
    stats: dict[str, SearchStats]

    def to_dict(self) -> dict[str, Any]:
        """Convert the benchmark result to a dictionary."""
        return asdict(self)


class BaselineManager:
    """Manage loading, saving, and comparing benchmark baselines."""

    def __init__(self, baseline_dir: Path):
        """Initialize the baseline manager with the specified directory."""
        self.baseline_dir = baseline_dir
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def load_baseline(self, name: str) -> dict[str, Any] | None:
        """Load a baseline by name from the baseline directory."""
        path = self.baseline_dir / f"{name}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def save_baseline(self, name: str, data: dict[str, Any]):
        """Save baseline data to the baseline directory."""
        path = self.baseline_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def compare(self, current: BenchmarkResult, baseline_name: str) -> dict[str, Any]:
        """Return a diff report comparing current results to a baseline."""
        baseline = self.load_baseline(baseline_name)
        if not baseline:
            return {"status": "NEW_BASELINE", "msg": "No previous baseline found."}

        diffs = {}
        for fen_name, current_stats in current.stats.items():
            if fen_name in baseline["stats"]:
                base_stats = baseline["stats"][fen_name]

                node_diff = current_stats.nodes - base_stats["nodes"]
                node_pct = (
                    (node_diff / base_stats["nodes"]) * 100
                    if base_stats["nodes"]
                    else 0
                )

                time_diff = current_stats.time_to_depth - base_stats["time_to_depth"]
                time_pct = (
                    (time_diff / base_stats["time_to_depth"]) * 100
                    if base_stats["time_to_depth"]
                    else 0
                )

                diffs[fen_name] = {
                    "nodes_diff": node_diff,
                    "nodes_pct": node_pct,
                    "time_diff_sec": time_diff,
                    "time_pct": time_pct,
                }

        return {"status": "COMPARED", "diffs": diffs}
