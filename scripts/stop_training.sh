#!/bin/bash
# ==============================================================================
# Training Stopper
# ==============================================================================
# Gracefully stops the training process
#
# Usage:
#   ./scripts/stop_training.sh [--force]
# ==============================================================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_ROOT/training.pid"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

FORCE=false
if [ "$1" == "--force" ]; then
    FORCE=true
fi

# ==============================================================================
# Functions
# ==============================================================================

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

stop_training() {
    if [ ! -f "$PID_FILE" ]; then
        print_error "No training process found (PID file missing)"
        exit 1
    fi

    PID=$(cat "$PID_FILE")

    if ! ps -p "$PID" > /dev/null 2>&1; then
        print_warning "Process not running (stale PID file)"
        rm "$PID_FILE"
        exit 0
    fi

    echo "Stopping training (PID: $PID)..."

    if [ "$FORCE" = true ]; then
        print_warning "Force killing process..."
        kill -9 "$PID" 2>/dev/null
        sleep 1
    else
        # Try graceful shutdown first
        echo "Sending SIGTERM (graceful shutdown)..."
        kill "$PID" 2>/dev/null

        # Wait up to 30 seconds for graceful shutdown
        for i in {1..30}; do
            if ! ps -p "$PID" > /dev/null 2>&1; then
                break
            fi
            echo -n "."
            sleep 1
        done
        echo ""

        # Check if still running
        if ps -p "$PID" > /dev/null 2>&1; then
            print_warning "Graceful shutdown failed. Force killing..."
            kill -9 "$PID" 2>/dev/null
            sleep 1
        fi
    fi

    # Verify it stopped
    if ps -p "$PID" > /dev/null 2>&1; then
        print_error "Failed to stop training process"
        exit 1
    else
        print_success "Training stopped successfully"
        rm "$PID_FILE"

        echo ""
        echo "Training was stopped. The last checkpoint is saved."
        echo "To resume training, edit scripts/full_pipeline_sft_grpo_training.py:"
        echo "  Set CONTINUE_FROM to the last checkpoint episode number"
        echo ""
        echo "Then restart with: ./scripts/start_training.sh"
    fi
}

# ==============================================================================
# Main
# ==============================================================================

echo "============================================================================="
echo "  Stop Training"
echo "============================================================================="
echo ""

if [ "$FORCE" = true ]; then
    print_warning "Force mode enabled (immediate kill)"
fi

stop_training
