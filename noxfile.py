"""Nox automation orchestrating uv-managed linting, typing, and tests."""

from __future__ import annotations

try:
    import nox
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Nox must be available in the developer environment to run automation sessions."
    ) from exc

PYTHON_VERSIONS = ["3.11"]
SOURCE_PATHS = ["src", "tests", "noxfile.py"]


def uv_run(
    session: nox.Session,
    *,
    command: list[str],
    extras: list[str] | None = None,
    with_packages: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Execute a command through uv so dependency resolution stays consistent.

    Args:
        session: Active nox session so we can delegate subprocess execution.
        command: Command and arguments to forward to ``uv run``.
        extras: Optional extras (e.g., ``dev``) that uv should install.
        with_packages: Additional packages to provision just for this command.
        env: Extra environment variables to expose to the subprocess.
    """
    extras_list = extras or []
    with_list = with_packages or []
    python_target = str(session.python or PYTHON_VERSIONS[0])

    uv_cmd: list[str] = ["uv", "run", "--python", python_target]

    for extra in extras_list:
        uv_cmd.extend(["--extra", extra])

    for package in with_list:
        uv_cmd.extend(["--with", package])

    uv_cmd.extend(command)
    session.run(*uv_cmd, external=True, env=env)


@nox.session(python=PYTHON_VERSIONS, venv_backend="none")
def tests(session: nox.Session) -> None:
    """Execute the pytest suite via uv to mirror the CI environment."""
    pytest_args = list(session.posargs) or ["tests"]
    uv_run(session, command=["pytest", *pytest_args], extras=["dev"])


@nox.session(python=PYTHON_VERSIONS, venv_backend="none")
def lint(session: nox.Session) -> None:
    """Enforce Ruff lint and format policies using the uv-managed toolchain."""
    uv_run(session, command=["ruff", "check", *SOURCE_PATHS], extras=["dev"])
    uv_run(
        session,
        command=["ruff", "format", "--check", *SOURCE_PATHS],
        extras=["dev"],
    )


@nox.session(name="types", python=PYTHON_VERSIONS, venv_backend="none")
def types(session: nox.Session) -> None:
    """Run the strict mypy configuration enforced in pyproject.toml."""
    uv_run(session, command=["mypy", "src"], extras=["dev"])


@nox.session(python=PYTHON_VERSIONS, venv_backend="none")
def pylance(session: nox.Session) -> None:
    """Mirror Pylance analysis by invoking the Pyright CLI through uv."""
    uv_run(
        session,
        command=["pyright", "src"],
        extras=["dev"],
        with_packages=["pyright"],
    )
