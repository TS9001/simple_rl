#!/bin/bash
# Run tests for Simple RL

set -e

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo "Setting up environment first..."
    ./scripts/setup.sh
fi

source .venv/bin/activate

# Parse test type
TEST_TYPE="${1:-all}"

case "$TEST_TYPE" in
    grpo)
        echo "Running GRPO tests..."
        pytest tests/test_grpo*.py -v
        ;;
    quick)
        echo "Running quick tests..."
        pytest -m "not slow" -v
        ;;
    coverage)
        echo "Running tests with coverage..."
        pytest --cov=simple_rl --cov-report=term-missing
        ;;
    all)
        echo "Running all tests..."
        pytest -v
        ;;
    *)
        echo "Usage: $0 [grpo|quick|coverage|all]"
        echo ""
        echo "  grpo     - Run only GRPO tests"
        echo "  quick    - Skip slow tests"
        echo "  coverage - Generate coverage report"
        echo "  all      - Run all tests (default)"
        exit 1
        ;;
esac