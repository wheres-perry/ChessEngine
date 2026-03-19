# Development Guide

## Branching Model

```
feature/*  →  dev  →  main (prod)
```

- **`feature/*`** — all day-to-day work happens here
- **`dev`** — integration branch; PRs from feature branches merge here
- **`main`** — production; only `dev → main` PRs, gated by benchmark comparison + sanitizers

## Quick Start

```bash
# Install everything (Python deps + C++ extension, editable):
uv sync --group dev

# Run the default CI gate locally (lint + types + fast tests):
uv run nox

# Run the full test suite:
uv run nox -s tests_all

# Run just one category:
pytest tests/search -v
```

## Build System

The C++ extension is built with **scikit-build-core** + **CMake** (replaced the legacy `setup.py`).

| File                              | Role                                                                 |
| --------------------------------- | -------------------------------------------------------------------- |
| `pyproject.toml` `[build-system]` | Declares `scikit-build-core` + `pybind11` as build deps              |
| `CMakeLists.txt`                  | Compiles `chess_engine_core` pybind11 module from `src/engine/_cpp/` |

### How it works

- `uv sync --group dev` triggers an **editable inplace build**: CMake compiles the `.so` directly into `src/engine/_core/`.
- A `.pth` file adds `/workspace/src` to `sys.path`, so imports use `from engine._core import chess_engine_core`.
- For wheel builds (non-editable), CMake's `install()` places the `.so` in the wheel at `engine/_core/`.

### Rebuilding after C++ changes

```bash
uv sync --group dev   # re-runs CMake inplace
```

### Import convention

All imports use `engine.*` (not `src.engine.*`). The `src/` is a layout directory, not part of the package name.

## Running Tests

### Via Nox (recommended — mirrors CI)

| Command                    | What it runs                             |
| -------------------------- | ---------------------------------------- |
| `uv run nox`               | Default: `lint` + `types` + `tests_fast` |
| `uv run nox -t safe`       | All fast sessions                        |
| `uv run nox -t heavy`      | Full tests + benchmarks + sanitizers     |
| `uv run nox -s tests_fast` | unit + smoke + search + evaluators       |
| `uv run nox -s tests_full` | parity + chess puzzles                   |
| `uv run nox -s tests_all`  | Everything                               |
| `uv run nox -s syzygy`     | Download 3-4-5 piece Syzygy tablebases   |
| `uv run nox -s benchmarks` | Full benchmarks with JSON + autosave     |
| `uv run nox -s sanitizers` | Recompile C++ with ASAN, run tests       |

### Via pytest directly

```bash
pytest tests/unit tests/search -v          # fast subset
pytest tests/ -v                           # everything
pytest tests/chess -v                      # just puzzles
pytest -m "not benchmark and not slow" -v  # skip slow stuff
pytest tests/benchmarks --benchmark-only   # benchmarks only
```

### Markers

Tests are auto-tagged by directory (see `tests/conftest.py`). Registered markers:

| Marker      | Applied to          | Purpose                         |
| ----------- | ------------------- | ------------------------------- |
| `slow`      | `tests/smoke/`      | Longer-running smoke tests      |
| `benchmark` | `tests/benchmarks/` | pytest-benchmark tests          |
| `parity`    | `tests/parity/`     | C++ vs python-chess correctness |
| `chess`     | `tests/chess/`      | Puzzle / tactic correctness     |

Use `-m` to filter: `pytest -m "not slow and not benchmark"`.

## Test Directory Structure

```
tests/
├── conftest.py              ← shared fixtures, auto-marker hook
├── helpers.py               ← test utilities (not collected)
│
├── unit/                    ← fast, isolated, no search
│   ├── test_config_robustness.py
│   ├── test_config_deps.py        ← config dependency validation
│   └── test_core_engine.py
│
├── search/                  ← minimax, TT, zobrist, ordering
│   ├── test_minimax.py
│   ├── test_move_ordering.py
│   ├── test_transposition_table.py
│   └── test_zobrist.py
│
├── evaluators/              ← component & factory tests
│   ├── conftest.py          ← evaluator fixtures
│   └── test_pst_tables.py
│
├── chess/                   ← puzzle / tactics framework
│   ├── conftest.py          ← load_fen_file() helper
│   ├── data/
│   │   └── mate_in_1.fen
│   └── test_mate_puzzles.py
│
├── parity/                  ← C++ board vs python-chess
│   ├── data/fens.txt
│   ├── data/golden/         ← regression snapshots
│   ├── test_golden.py
│   ├── test_legal_moves.py
│   └── test_random_games.py
│
├── benchmarks/              ← persistent benchmarks
│   ├── infrastructure.py
│   ├── test_performance.py
│   └── test_search_metrics.py
│
└── smoke/                   ← quick speed gates
    └── test_speed.py
```

### Adding Tests

- Drop a `test_*.py` file in the appropriate directory — the root conftest auto-tags it.
- **New puzzle category:** add a `.fen` file to `tests/chess/data/`, load with `load_fen_file("filename.fen")` from `tests/chess/conftest.py`. Each line: `<FEN> <expected_uci_move>`.
- **New parity regression:** use `pytest-regressions` — golden YAML files live in `tests/parity/data/golden/`.
- **Evaluator architecture** uses a factory pattern: `EvaluatorFactory.create(config.evaluation)` assembles a `CompositeEvaluator` from individual `EvalComponent` objects (Material, PST, PawnStructure, Mobility, KingSafety). The `game_stage_conscious` flag enables phase-aware weighting across all components.

## Benchmarks (Persistent)

Results are saved to `.benchmarks/` at the repo root on every `--benchmark-autosave` run.

```bash
# Run and save a baseline:
pytest tests/benchmarks --benchmark-only --benchmark-autosave

# Compare against a previous run:
pytest tests/benchmarks --benchmark-only --benchmark-compare=0001

# Full benchmark session (JSON output for CI):
uv run nox -s benchmarks
```

In CI:

- **Merges to `dev`** — benchmarks run, results are uploaded as artifacts (90-day retention) and published to a GitHub Pages dashboard.
- **PRs into `main`** — benchmarks run and are compared against the `dev` baseline. The PR **fails if any benchmark regresses >15%**.

## CI/CD Pipeline

Defined in `.github/workflows/ci.yml`. Five gates, progressively stricter:

```
feature/* push     →  Gate 1: lint + types + fast tests
PR into dev/main   →  Gate 2: + full test suite
dev push (merge)   →  Gate 3: + benchmark persistence & dashboard
PR into main       →  Gate 4: + benchmark regression check (fail >15%)
                      Gate 5: + ASAN/UBSAN sanitizers
```

| Gate | Trigger                               | Jobs                                     |
| ---- | ------------------------------------- | ---------------------------------------- |
| 1    | Every push / PR                       | `lint-and-types` → `tests-fast`          |
| 2    | PRs into dev/main, pushes to dev/main | `tests-full`                             |
| 3    | Pushes to `dev` only                  | `benchmarks` (autosave + dashboard)      |
| 4    | PRs into `main` only                  | `benchmark-compare` (fail on regression) |
| 5    | PRs into `main` only                  | `sanitizers` (ASAN/UBSAN)                |

Duplicate runs on the same ref are cancelled automatically via `concurrency`.

## Syzygy Endgame Tablebases

The engine ships a download script for **3-4-5 piece Syzygy tablebases** (~1 GB).
These files are **not stored in Git** — they are downloaded on-demand from the
[Lichess open-source mirror](https://tablebase.lichess.ovh/) and cached locally.

### Quick Start (Local)

```bash
# Download tables to the default location (data/syzygy/):
python scripts/download_syzygy.py

# Or use nox:
uv run nox -s syzygy

# Download to a custom path:
python scripts/download_syzygy.py --path /mnt/ssd/syzygy

# Override via environment variable:
export SYZYGY_PATH=/mnt/ssd/syzygy
python scripts/download_syzygy.py

# Verify existing tables without downloading:
python scripts/download_syzygy.py --verify-only
```

### How It Works

| Component                    | What it does                                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------------------ |
| `scripts/download_syzygy.py` | Scrapes the Lichess mirror, downloads missing files with parallel workers, validates sizes       |
| `SYZYGY_PATH` env var        | Overrides the default `data/syzygy/` path used by the script and `constants.DEFAULT_SYZYGY_PATH` |
| `.gitignore` entry           | `data/syzygy/` is excluded from version control                                                  |

### Path Resolution Order

1. **Explicit argument** - `--path` flag
2. **`SYZYGY_PATH` environment variable**
3. **Project default** - `data/syzygy/`

### In CI (GitHub Actions)

The `tests-all` job in `.github/workflows/ci.yml` uses `actions/cache@v4` to
persist the `data/syzygy/` directory across runs. On a cache miss, the download
script runs automatically. The cache key is `syzygy-3-4-5-v1` — bump to `v2`
if you ever change the file set.

## Nox Sessions

All sessions use `uv` as the venv backend. The shared `_install()` helper runs `uv sync --frozen --group dev`.

| Session            | Tag   | Description                                    |
| ------------------ | ----- | ---------------------------------------------- |
| `lint`             | safe  | ruff check + format, clang-format on C++       |
| `types`            | safe  | mypy on `src/`, `noxfile.py`                   |
| `tests_fast`       | safe  | unit, smoke, search, evaluators                |
| `benchmarks_smoke` | safe  | Single benchmark file, 1 round                 |
| `tests_full`       | heavy | parity, chess                                  |
| `tests_all`        | heavy | Entire `tests/` directory                      |
| `syzygy`           | heavy | Download 3-4-5 piece Syzygy endgame tablebases |
| `benchmarks`       | heavy | All benchmarks, JSON output, autosave          |
| `sanitizers`       | heavy | Rebuild C++ with ASAN, run unit+smoke+search   |

## Zobrist Hashing

Zobrist hashing is implemented entirely in C++ (`src/engine/_cpp/zobrist_keys.hpp`) and exposed to Python via pybind11 as `chess_engine_core.Zobrist`.

- Keys are generated at **compile time** using a `constexpr` SplitMix64 PRNG seeded from a fixed constant - no runtime initialization cost.
- Tables are **cache-aligned** (`alignas(64)`) for L1 performance.
- `hash_board(board)` - full board re-hash (used after `set_fen`).
- `make_move_hash(board, move)` - O(1) incremental hash for a move (does not push the move).
- `make_null_move_hash(board)` - O(1) incremental hash for a null move (toggles side-to-move and removes en-passant square).
- `get_current_hash()` / `set_current_hash(h)` - read/write the stored hash value.

The Python wrapper lives in `src/engine/search/zobrist.py` and simply re-exports `Zobrist = chess_engine_core.Zobrist`.

Microbenchmarks are in `scripts/bench_zobrist.py`:

```bash
uv run python scripts/bench_zobrist.py
```

## Config System

`EngineConfig` wraps `SearchConfig` + `EvaluationConfig`. Key design decisions:

- **Minimax and IDDFS are always on** — no toggle flags.
- **Zobrist hashing + transposition table** are combined under `use_transposition_table`.
- Config validates dependencies on construction (`__post_init__`). For example, `use_pvs=True` requires `use_alpha_beta=True`, `use_killer_moves=True` requires `use_move_ordering=True`, etc.
- `ConfigSolver` performs a comprehensive validation pass (driven by `ConfigSolverRules`) before the engine is constructed, raising `ConfigSolverError` on violations.

### Supported Feature Matrix

Only the following feature surfaces are supported and should be used in configs/tests.

| Feature                                   | Config key(s)                                                      |
| ----------------------------------------- | ------------------------------------------------------------------ |
| Piece-Square Tables                       | `evaluation.use_pst`                                               |
| Pawn Structure Tables                     | `evaluation.use_pawn_structure` (requires PST)                     |
| Mobility Heuristics                       | `evaluation.use_mobility`                                          |
| King Safety Heuristics                    | `evaluation.use_king_safety`                                       |
| Game Stage Conscious (GSC)                | `evaluation.game_stage_conscious`                                  |
| Hash Move Ordering                        | `search.use_hash_move_ordering`                                    |
| MVV-LVA                                   | `search.use_mvv_lva`                                               |
| Static Exchange Evaluation (SEE) Ordering | `search.use_see_ordering`                                          |
| Killer Heuristic                          | `search.use_killer_moves`                                          |
| History / Countermove Heuristics          | `search.use_history_heuristic`, `search.use_countermove_heuristic` |
| Principal Variation Search (PVS)          | `search.use_pvs`                                                   |
| Aspiration Windows                        | `search.use_aspiration_windows`                                    |
| Internal Iterative Deepening (IID)        | `search.use_iid`                                                   |
| Late Move Reductions (LMR)                | `search.use_lmr`                                                   |
| Check Extensions                          | `search.use_check_extensions`                                      |
| Null Move Pruning (NMP)                   | `search.use_null_move_pruning`                                     |
| Futility Pruning (Standard)               | `search.use_futility_pruning`                                      |
| Futility Pruning (Extended)               | `search.use_extended_futility_pruning`                             |
| Futility Pruning (Reverse)                | `search.use_reverse_futility_pruning`                              |
| Delta Pruning                             | `search.use_delta_pruning`                                         |
| SEE in Quiescence Search                  | `search.use_see_pruning_in_qs`                                     |
| TT Aging / Eviction                       | `search.use_tt_aging`                                              |

Note: foundational toggles such as `search.use_alpha_beta`, `search.use_move_ordering`, `search.use_transposition_table`, and `search.use_quiescence_search` remain first-class because they are dependencies for multiple listed features.

## C++ Board API Notes

The C++ board (`engine._core.chess_engine_core`) differs from python-chess in a few ways:

| python-chess                    | C++ Board                     | Notes                                           |
| ------------------------------- | ----------------------------- | ----------------------------------------------- |
| `chess.Board("fen")`            | `chess.Board.from_fen("fen")` | Constructor doesn't accept FEN                  |
| `board.turn == chess.WHITE`     | `bool(board.turn)`            | `turn` is `bool` (True=white), not a Color enum |
| `chess.D4`                      | `27` (int)                    | No named square constants                       |
| `board.legal_moves` (generator) | `board.legal_moves` (list)    | Already a list, don't call it                   |
| `str(move)` → `"e2e4"`          | `move.uci()` → `"e2e4"`       | `str()` gives `<Move e2e4>`                     |
| `board.pieces(PAWN, WHITE)`     | `board.pieces(PAWN, WHITE)`   | Returns list of square ints                     |
| `chess.square(file, rank)`      | `rank * 8 + file`             | No `square()` helper                            |

## Tool Configuration

All tool config lives in `pyproject.toml` (single source of truth):

- **Ruff** — `[tool.ruff]` and `[tool.ruff.lint]` (replaces deleted `ruff.toml`)
- **Mypy** — `[tool.mypy]`
- **Pytest** — `[tool.pytest.ini_options]` (strict-markers, coverage)
- **Benchmark** — `[tool.pytest.ini_options.benchmark]`

Dev dependencies use PEP 735 dependency groups: `[dependency-groups] dev = [...]`.

## Pre-commit

`.pre-commit-config.yaml` runs:

- `ruff` (check + format)
- `clang-format` (C++ files)
- `pre-commit-hooks` (trailing whitespace, EOF fixer, YAML check, large file check)

Install: `pre-commit install`. Runs automatically on `git commit`.
