[] Fixbroken benchmark in nox, "nox > Command pytest --benchmark-only --benchmark-json=output.json --benchmark-autosave tests/benchmarks failed with exit code -6"
[] Dependency Resolution with z3 solver

[] This broken thing:
    session.log("Running tests with ASAN active...")
    session.run(
        "pytest",
        "tests/unit",
        "tests/smoke",
        "tests/search",
        "-v",
        env=run_env,
    )

