#!/bin/bash
set -e

IMAGE_NAME="chess-engine"
VERBOSE=false
TEST_PATH=""
PYTEST_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [test_path] [-verbose] [pytest_args...]"
            echo "  test_path: Path to test file or directory (default: tests/)"
            echo "  -verbose: Run pytest with -vv for verbose output"
            echo "  pytest_args: Additional arguments to pass to pytest"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all tests"
            echo "  $0 tests/search/zobrist_test.py      # Run specific test"
            echo "  $0 -verbose                          # Run all tests with verbose output"
            echo "  $0 tests/search/ -verbose -k 'test_aging'"
            exit 0
            ;;
        -*)
            # This is a pytest argument
            PYTEST_ARGS+=("$1")
            shift
            ;;
        *)
            # This is the test path (if not already set)
            if [[ -z "$TEST_PATH" ]]; then
                TEST_PATH="$1"
            else
                # Additional pytest arguments
                PYTEST_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# Set default test path if not provided
if [[ -z "$TEST_PATH" ]]; then
    TEST_PATH="tests/"
fi

# Create cache directory if it doesn't exist (suppress error if it exists)
mkdir -p .docker-cache

# Build the last stage only (test-runtime) with cache
echo "Building last stage (test-runtime) of Docker image '$IMAGE_NAME'..."
docker buildx build \
    --target test-runtime \
    --cache-from type=local,src=.docker-cache \
    --cache-to type=local,dest=.docker-cache \
    -t "$IMAGE_NAME" \
    .

# Prepare pytest command - run pytest directly, not through poetry
PYTEST_CMD=(
    docker run --rm -it 
    -e PYTHONPATH=/app 
    "$IMAGE_NAME" 
    python -m pytest 
    "$TEST_PATH"
)

# Add verbose flag
if [[ "$VERBOSE" == true ]]; then
    PYTEST_CMD+=("-vv")
else
    PYTEST_CMD+=("-v")  # Default to -v as in your original script
fi

# Add any additional pytest args
if [[ ${#PYTEST_ARGS[@]} -gt 0 ]]; then
    PYTEST_CMD+=("${PYTEST_ARGS[@]}")
fi

echo "Running pytest inside container '$IMAGE_NAME' with args: ${PYTEST_CMD[@]:6}..."
echo "-------------------------"

# Execute the pytest command
"${PYTEST_CMD[@]}"

echo "-------------------------"
echo "Test run complete."
