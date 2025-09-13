#!/bin/bash

# Script to run tests for Simple RL

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Running setup first...${NC}"
    ./setup.sh
fi

# Activate virtual environment
source .venv/bin/activate

echo -e "${GREEN}"
echo "======================================================"
echo "            Simple RL - Test Suite                   "
echo "======================================================"
echo -e "${NC}"
echo ""

# Parse arguments
TEST_TYPE="${1:-all}"
VERBOSE="${2:-}"

case "$TEST_TYPE" in
    grpo)
        echo -e "${BLUE}Running GRPO tests...${NC}"
        pytest tests/test_grpo*.py -v
        ;;
    quick)
        echo -e "${BLUE}Running quick tests (no slow markers)...${NC}"
        pytest -m "not slow" -v
        ;;
    coverage)
        echo -e "${BLUE}Running tests with coverage...${NC}"
        pytest --cov=simple_rl --cov-report=term-missing
        ;;
    all)
        echo -e "${BLUE}Running all tests...${NC}"
        pytest -v
        ;;
    *)
        echo "Usage: $0 [grpo|quick|coverage|all] [options]"
        echo ""
        echo "Test types:"
        echo "  grpo     - Run only GRPO-related tests"
        echo "  quick    - Run quick tests (skip slow ones)"
        echo "  coverage - Run with coverage report"
        echo "  all      - Run all tests (default)"
        exit 1
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo ""
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi