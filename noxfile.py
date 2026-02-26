"""Nox sessions for the chess engine CI pipeline.

Sessions are tagged for tiered execution:
  safe   -  lint, types, fast tests  (every push)
  heavy  -  full tests, benchmarks, sanitizers  (PRs / main)

Usage:
  uv run nox              # default: lint + types + tests_fast
  uv run nox -t safe      # all safe sessions
  uv run nox -t heavy     # all heavy sessions
  uv run nox -s tests_all # complete pytest run
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import TYPE_CHECKING

import nox

if TYPE_CHECKING:
    from nox.sessions import Session

# --- Global options ---
nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "types", "tests_fast"]


# --- Helpers ---
def _install(session: Session) -> None:
    """Shared install step: sync frozen lockfile with dev deps.

    --active tells uv to use the currently activated venv (the nox-managed
    one) rather than the project's .venv, which it would otherwise prefer.
    """
    session.run_install(
        "uv",
        "sync",
        "--active",
        "--frozen",
        "--group",
        "dev",
    )


# --- Safe sessions (every push) ---
@nox.session(tags=["safe"])
def lint(session: Session) -> None:
    """Fast linting and formatting checks."""
    _install(session)

    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")

    # C++ formatting (Google Style)
    cpp_files: list[str] = []
    for root, _, files in os.walk("src"):
        cpp_files.extend(
            os.path.join(root, f) for f in files if f.endswith((".cpp", ".hpp"))
        )
    if cpp_files:
        session.run("clang-format", "--dry-run", "--Werror", *cpp_files, external=True)


@nox.session(tags=["safe"])
def types(session: Session) -> None:
    """Static type checking with mypy."""
    _install(session)
    session.run("mypy", "src", "noxfile.py")


@nox.session(tags=["safe"])
def tests_fast(session: Session) -> None:
    """Unit + smoke + search + evaluator tests (fast feedback loop)."""
    _install(session)
    session.run(
        "pytest",
        "tests/unit",
        "tests/smoke",
        "tests/search",
        "tests/evaluators",
        "-v",
    )


@nox.session(tags=["safe"])
def benchmarks_smoke(session: Session) -> None:
    """Lightweight benchmark smoke check."""
    _install(session)
    session.run(
        "pytest",
        "--benchmark-only",
        "--benchmark-min-rounds=1",
        "tests/benchmarks/test_search_metrics.py",
    )


# --- Heavy sessions (PRs / main) ---
@nox.session(tags=["heavy"])
def syzygy(session: Session) -> None:
    """Ensure 3-4-5 piece Syzygy tablebases are downloaded."""
    _install(session)
    session.run("python", "scripts/download_syzygy.py")


@nox.session(tags=["heavy"])
def tests_full(session: Session) -> None:
    """Integration + parity + chess puzzle tests."""
    _install(session)
    session.run(
        "pytest",
        "tests/parity",
        "tests/chess",
        "-v",
    )


@nox.session(tags=["heavy"])
def tests_all(session: Session) -> None:
    """Complete pytest run across every test folder."""
    _install(session)
    session.run("pytest", "-v")


@nox.session(tags=["heavy"])
def benchmarks(session: Session) -> None:
    """Scientific benchmarks with JSON output for the dashboard."""
    _install(session)
    session.run(
        "pytest",
        "--benchmark-only",
        "--benchmark-json=output.json",
        "--benchmark-autosave",
        "tests/benchmarks",
    )


@nox.session(tags=["heavy"])
def sanitizers(session: Session) -> None:
    """Recompile C++ extensions with ASAN/UBSAN and run checks."""
    _install(session)

    # Locate libasan
    libasan_paths = sorted(glob.glob("/usr/lib/x86_64-linux-gnu/libasan.so*"))
    ld_preload = libasan_paths[0] if libasan_paths else ""
    if ld_preload:
        session.log(f"Found libasan: {ld_preload}")
    else:
        session.log("Could not find libasan.so  -  ASan may fail to load.")

    # Compiler flags
    build_env = {
        "CFLAGS": "-fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer",
        "CXXFLAGS": "-fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer",
        "LDFLAGS": "-fsanitize=address,undefined",
    }

    # Runtime flags
    run_env = {
        **build_env,
        "ASAN_OPTIONS": "detect_leaks=1",
        "LSAN_OPTIONS": "suppressions=tools/lsan.supp",
        **({"LD_PRELOAD": ld_preload} if ld_preload else {}),
    }

    # 1. Clean build artifacts
    session.log("Cleaning build artifacts for sanitizer run...")
    shutil.rmtree("build", ignore_errors=True)
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith((".so", ".pyd")):
                os.remove(os.path.join(root, f))

    # 2. Recompile with sanitizer flags
    session.log("Recompiling C++ extensions with ASAN...")
    session.run_install(
        "uv",
        "sync",
        "--frozen",
        "--reinstall-package",
        "chessengine",
        "--group",
        "dev",
        env=build_env,
    )

    # 3. Run tests under sanitizers
    session.log("Running tests with ASAN active...")
    session.run(
        "pytest",
        "tests/unit",
        "tests/smoke",
        "tests/search",
        "-v",
        env=run_env,
    )
