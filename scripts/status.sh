#!/bin/bash
# ==============================================================================
# Training Status Checker
# ==============================================================================
# Shows current training status, metrics, and resource usage
#
# Usage:
#   ./scripts/status.sh
# ==============================================================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_ROOT/training.pid"
LOG_FILE="$PROJECT_ROOT/training.log"
PROGRESS_FILE="$PROJECT_ROOT/logs/progress.json"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ==============================================================================
# Functions
# ==============================================================================

print_header() {
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}  Training Status${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
}

check_process() {
    echo -e "${BLUE}Process Status:${NC}"
    if [ ! -f "$PID_FILE" ]; then
        echo -e "  ${RED}✗ No training running${NC}"
        return 1
    fi

    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓ Training RUNNING${NC}"
        echo -e "  PID: $PID"
        echo ""
        echo "  Process details:"
        ps -p "$PID" -o pid,ppid,%cpu,%mem,etime,cmd | tail -n +2 | sed 's/^/    /'
        return 0
    else
        echo -e "  ${RED}✗ Training NOT running (stale PID file)${NC}"
        return 1
    fi
}

show_progress() {
    echo ""
    echo -e "${BLUE}Training Progress:${NC}"

    if [ -f "$PROGRESS_FILE" ]; then
        if command -v python3 &> /dev/null; then
            python3 "$SCRIPT_DIR/monitor_training.py"
        else
            # Fallback to basic JSON display
            cat "$PROGRESS_FILE" | grep -E '"(status|current_episode|total_episodes|progress_percent|latest_metrics)"' | sed 's/^/  /'
        fi
    else
        echo -e "  ${YELLOW}⚠ No progress file found yet${NC}"
    fi
}

show_recent_log() {
    echo ""
    echo -e "${BLUE}Recent Log Output (last 15 lines):${NC}"
    if [ -f "$LOG_FILE" ]; then
        tail -15 "$LOG_FILE" | sed 's/^/  /'
    else
        echo -e "  ${YELLOW}⚠ No log file found yet${NC}"
    fi
}

show_system_resources() {
    echo ""
    echo -e "${BLUE}System Resources:${NC}"

    # CPU and Memory
    echo "  CPU & Memory:"
    top -b -n 1 | grep -A 5 "^%Cpu" | head -6 | sed 's/^/    /'

    # GPU (if available)
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "  GPU:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader | sed 's/^/    /'
    fi

    # Disk space
    echo ""
    echo "  Disk Space (checkpoints):"
    df -h "$PROJECT_ROOT" | tail -1 | awk '{print "    Used: "$3" / "$2" ("$5" full)"}'
}

show_checkpoints() {
    echo ""
    echo -e "${BLUE}Checkpoints:${NC}"
    CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/grpo_qwen_math"

    if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR 2>/dev/null)" ]; then
        echo "  Found $(ls -1 "$CHECKPOINT_DIR" | wc -l) checkpoint(s)"
        echo ""
        echo "  Latest checkpoints:"
        ls -lht "$CHECKPOINT_DIR" | head -6 | tail -5 | awk '{print "    "$9" - "$5" ("$6" "$7" "$8")"}'
    else
        echo -e "  ${YELLOW}⚠ No checkpoints found yet${NC}"
    fi
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    print_header
    echo ""

    if ! check_process; then
        echo ""
        echo "Start training with: ./scripts/start_training.sh"
        exit 1
    fi

    show_progress
    show_recent_log
    show_system_resources
    show_checkpoints

    echo ""
    echo -e "${BLUE}=============================================================================${NC}"
    echo ""
    echo "Commands:"
    echo "  Monitor live:     tail -f $LOG_FILE"
    echo "  Watch progress:   python3 $SCRIPT_DIR/monitor_training.py --watch"
    echo "  Stop training:    $SCRIPT_DIR/stop_training.sh"
    echo ""
}

main
