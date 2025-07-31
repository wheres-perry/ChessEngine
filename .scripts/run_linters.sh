#!/bin/bash
# filepath: /workspace/.scripts/run_linters.sh

set -eo pipefail

echo "===== Running Python Linters ====="
EXIT_CODE=0

# Function to run a command and track exit code
run_command() {
    echo -e "\n>>> Running $1"
    eval "$2"
    CMD_EXIT=$?
    if [ $CMD_EXIT -ne 0 ]; then
        echo -e "✗ $1 failed with exit code $CMD_EXIT"
        EXIT_CODE=1
    else
        echo -e "✓ $1 passed"
    fi
    return $CMD_EXIT
}

# Find poetry
POETRY_BIN=$(which poetry)

# Clean Python cache before running linters
echo ">>> Cleaning Python cache files"
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf .mypy_cache/ 2>/dev/null || true

# Run Ruff Linter
run_command "Ruff Linter" "$POETRY_BIN run ruff check src/ --fix" || true

# Run Ruff Formatter
run_command "Ruff Formatter" "$POETRY_BIN run ruff format src/" || true

# Run MyPy - override PYTHONPATH to empty, set MYPYPATH, and target source paths directly
run_command "MyPy Type Checker" "PYTHONPATH=/workspace $POETRY_BIN run mypy --config-file pyproject.toml src" || true

# Run Pylint with PYTHONPATH set for module discovery
run_command "Pylint" "PYTHONPATH=/workspace $POETRY_BIN run pylint src" || true

# Summary
echo -e "\n===== Linting Summary ====="
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "✓ All linters passed"
else
    echo -e "✗ Some linters reported issues"
    echo -e "⛔ Commit prevented due to linting errors"
fi

exit $EXIT_CODE
