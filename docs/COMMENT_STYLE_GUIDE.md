# ChessEngine Comment Style Guide

This guide establishes the standard for commenting and documenting code across the `ChessEngine` project. As a high-performance, architecturally complex project, the goal of our comments is to explain **why** and **how**, never *what*.

## 1. Core Philosophy

*   **No "What" Comments:** Do not translate code into English. Assume the reader is a competent engineer.
    *   *Bad:* `// Shift bitboard left by 9`
    *   *Good:* `// Shift by 9 to compute North-West attacks; mask with ~FILE_H to prevent wrapping around the board edge.`
*   **"Why" and "How" Only:** Explain algorithmic choices, mathematical invariants, business logic, and edge cases.
*   **Visual Documentation:** Chess relies on 2D spatial relationships and bitwise operations. Use ASCII art liberally to visually represent tensor shapes, bitboard masks, and board states.

## 2. Python Standards: Google Style Docstrings

All Python modules, classes, and public functions **must** use the [Google Style Docstring](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) format.

This standard is automatically enforced via `ruff` in our CI pipeline using the `pydocstyle` (prefix `D`) rules.

### Structure of a Function Docstring

1.  **Summary Line:** A one-line summary ending in a period, written in the imperative mood (e.g., "Extract...", "Compute...", not "Extracts...").
2.  **Extended Description:** (Optional) A detailed explanation of the logic, performance characteristics, or invariants.
3.  **Args:** A list of arguments and their descriptions. Types are omitted here as they are covered by Python type hints.
4.  **Returns:** A description of the return value.
5.  **Raises:** (Optional) A list of exceptions that are explicitly raised.

### Example

```python
def extract_cnn(board: core.Board) -> np.ndarray:
    """Extract a spatial tensor representation of the chess board for CNN inference.

    Treats the board as an image, encoding piece types, colors, side-to-move, 
    and castling rights into distinct channels. The output is memory-mapped 
    directly from the C++ core for zero-copy performance.

    Args:
        board: The current C++ Bitboard state.

    Returns:
        A 3D numpy array of shape (17, 8, 8) of type float32. 
        Channels 0-11 represent pieces, 12 is STM, 13-16 are castling rights.
    """
    pass
```

## 3. C++ Standards: Doxygen Header Documentation

C++ code enforces a strict separation between the **API Contract** and the **Implementation**.

*   **Header Files (`.hpp`):** Must contain all documentation describing what a function does, its parameters, and its return values using Doxygen-style `///` or `/** ... */` syntax.
*   **Implementation Files (`.cpp`):** Should *only* contain inline comments explaining tricky logic or algorithmic steps. Do not duplicate the header documentation.

### Example Header (`.hpp`)

```cpp
/**
 * @brief Computes the pseudo-legal moves for a sliding piece using magic bitboards.
 * 
 * This uses a precomputed hash table to bypass loop-based ray tracing, 
 * reducing move generation time to O(1).
 * 
 * @param sq The square of the sliding piece (0-63).
 * @param occupied The bitboard of all currently occupied squares.
 * @return A bitboard containing all attacked squares.
 */
[[nodiscard]] Bitboard get_ray_attacks(int sq, Bitboard occupied) noexcept;
```

### Example Implementation (`.cpp`)

```cpp
Bitboard get_ray_attacks(int sq, Bitboard occupied) noexcept {
    // Mask out the relevant occupancy for this square's rays
    Bitboard blockers = occupied & get_ray_mask(sq);
    
    // Hash the blockers to find the precomputed attack subset
    int magic_index = (blockers * MAGIC_NUMBERS[sq]) >> MAGIC_SHIFTS[sq];
    return ATTACK_TABLE[sq][magic_index];
}
```

## 4. Special Annotations

Use the following prefixes for inline comments to ensure they are caught by IDE highlighters and reviewers:

*   `// TODO(your-name):` - For planned features or known technical debt. Always attach a name to show ownership.
*   `// FIXME:` - For known bugs, edge cases, or broken logic that needs addressing before release.
*   `// NOTE:` - To call out unusual, unintuitive, or API-specific behavior (e.g., `// NOTE: Pybind11 takes ownership of this pointer`).
*   `// PERF:` - To explain code that sacrifices readability for the sake of strict performance optimization (common in move generation and search).

## 5. Automated Enforcement

To ensure these standards are maintained, the following `ruff` configuration is enforced in `pyproject.toml`:

*   `D100`, `D101`, `D102`, `D103`: Enforce docstrings on modules, classes, and public methods.
*   `D205`: Enforce that the summary line starts on the same line as the opening quotes.
*   `D417`: Enforce a blank line between the summary and the description.
*   `convention = "google"`: Strictly enforce the Google formatting style.
