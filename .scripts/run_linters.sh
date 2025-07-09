#!/bin/bash
# filepath: /workspace/.scripts/run_linters.sh

set -eo pipefail
export PYTHONPATH=/workspace

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

# Find poetry and the virtualenv
POETRY_BIN=$(which poetry)

# Install and run Ruff - without --user flag in virtualenv
run_command "Ruff Linter" "$POETRY_BIN run pip install ruff && $POETRY_BIN run ruff check src/ --fix" || true

# Run Ruff Formatter
run_command "Ruff Formatter" "$POETRY_BIN run ruff format src/" || true

# Run MyPy for type checking with proper PYTHONPATH
run_command "MyPy Type Checker" "PYTHONPATH=/workspace $POETRY_BIN run mypy --namespace-packages src/" || true

# Run Pylint for code quality
run_command "Pylint" "$POETRY_BIN run pylint src/" || true

# Summary
echo -e "\n===== Linting Summary ====="
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "✓ All linters passed"
else
    echo -e "✗ Some linters reported issues"
    echo -e "⛔ Commit prevented due to linting errors"
fi

# Return the actual exit code to prevent commits when there are errors
exit $EXIT_CODE