# ChessEngine Naming & Conventions Guide

This guide establishes the standard for naming variables, functions, classes, and tests across the `ChessEngine` project. Consistency is key to a maintainable and readable codebase.

## 1. Core Python Naming Conventions (PEP 8)

The project strictly follows [PEP 8](https://peps.python.org/pep-0008/) naming conventions, which are automatically enforced by the `pep8-naming` (`N`) rule set in `ruff`.

*   **Variables, Functions, and Methods:** `snake_case` (e.g., `calculate_halfkp`, `board_state`).
*   **Classes and Exceptions:** `PascalCase` / `UpperCamelCase` (e.g., `UCIHandler`, `InvalidMoveError`).
*   **Constants (Module-level):** `UPPER_SNAKE_CASE` (e.g., `STARTING_FEN`, `MAX_DEPTH`).
*   **Private/Internal Members:** Prefix with a single underscore (e.g., `_dispatch`, `_parse_san`). This signals that the member is an implementation detail and not part of the public API.

## 2. Test Suite Conventions (pytest)

The test suite is built on `pytest` and heavily utilizes its features. These conventions are partially enforced by the `flake8-pytest-style` (`PT`) rule set in `ruff`.

### A. Test File and Function Naming

*   **Files:** Must be prefixed with `test_` (e.g., `test_move_generation.py`).
*   **Functions:** Must be prefixed with `test_`. The name should clearly describe *what* is being tested and, ideally, the expected outcome or scenario.
    *   *Bad:* `test_board()`
    *   *Good:* `test_board_copy_preserves_en_passant_state()`
*   **Classes (for grouping):** Must be prefixed with `Test`. Use classes only to group related tests or share fixtures via setup methods. Do *not* use them if they don't provide grouping value.
    *   *Example:* `TestZobristIncrementalUpdates`

### B. Parameterization (`@pytest.mark.parametrize`)

When testing multiple edge cases of the same logic, **always** use parameterization rather than writing duplicate test functions.

*   **Argument Names:** Pass parameter names as a tuple of strings (enforced by `PT006`), not a comma-separated string.
*   **Clarity:** Always include a `description` or `scenario` string in your parameters so test failures are immediately understandable in the CI output.

```python
# GOOD
@pytest.mark.parametrize(
    ("fen", "expected_count", "description"),
    [
        ("4k3/8... w K - 0 1", 4, "Prevented castle due to block"),
        ("r3k3/8... b - - 0 1", 3, "No queen castle thru attacker"),
    ]
)
def test_move_generation_edge_cases(fen, expected_count, description):
    ...
```

### C. Fixtures

*   **Naming:** Fixtures should be named as nouns representing what they provide (e.g., `starting_board`, `mock_uci_stream`). Do not prefix them with `setup_` or `get_`.
*   **Scope:** Be mindful of fixture scope. Use `scope="session"` or `scope="module"` for expensive setups (like loading large PGN files) to keep the test suite fast.

## 3. C++ Naming Conventions (Google Style)

The C++ backend bindings follow a slightly modified Google C++ Style Guide, optimized for Python integration.

*   **Variables:** `snake_case` (e.g., `piece_bitboards`).
*   **Functions/Methods:** `snake_case` (e.g., `generate_legal_moves()`). *Note: This diverges from standard Google C++ (which uses CamelCase for functions) to ensure seamless 1:1 mapping when exposed to Python via PyBind11.*
*   **Classes/Structs:** `PascalCase` (e.g., `GraphData`, `StateInfo`).
*   **Constants/Macros:** `UPPER_SNAKE_CASE` (e.g., `NUM_PIECE_TYPES`).
*   **Private Members:** Suffix with an underscore (e.g., `cached_attacked_by_`).

## 4. Automated Enforcement

To ensure these standards are maintained, the following `ruff` configuration is active in `pyproject.toml`:

*   `N` (`pep8-naming`): Enforces standard Python naming (snake_case functions, PascalCase classes).
*   `PT` (`flake8-pytest-style`): Enforces pytest best practices (e.g., `PT006` for tuple parameterization).
*   `C/C++ Formatting`: `clang-format` is run in CI to enforce the C++ style.