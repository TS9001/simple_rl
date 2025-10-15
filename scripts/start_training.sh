#!/bin/bash
# ==============================================================================
# GRPO Training Launcher for Remote Servers
# ==============================================================================
# This script starts training with proper logging and background execution.
# Training continues even after SSH disconnect.
#
# Usage:
#   ./scripts/start_training.sh
#
# ==============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRAINING_SCRIPT="$SCRIPT_DIR/full_pipeline_sft_grpo_training.py"
LOG_DIR="$PROJECT_ROOT/logs"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
PID_FILE="$PROJECT_ROOT/training.pid"
LOG_FILE="$PROJECT_ROOT/training.log"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ==============================================================================
# Functions
# ==============================================================================

print_header() {
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}  GRPO Training Launcher${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "python3 not found. Please install Python 3."
        exit 1
    fi
    print_success "Python found: $(python3 --version)"
}

check_training_script() {
    if [ ! -f "$TRAINING_SCRIPT" ]; then
        print_error "Training script not found: $TRAINING_SCRIPT"
        exit 1
    fi
    print_success "Training script found"
}

check_existing_training() {
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            print_warning "Training already running with PID: $OLD_PID"
            echo ""
            echo "Options:"
            echo "  1. Stop existing training and start new one"
            echo "  2. Keep existing training and exit"
            echo "  3. View status of existing training"
            echo ""
            read -p "Enter choice (1/2/3): " choice

            case $choice in
                1)
                    print_info "Stopping existing training..."
                    kill "$OLD_PID" 2>/dev/null || true
                    sleep 2
                    if ps -p "$OLD_PID" > /dev/null 2>&1; then
                        print_warning "Graceful stop failed, force killing..."
                        kill -9 "$OLD_PID" 2>/dev/null || true
                        sleep 1
                    fi
                    print_success "Existing training stopped"
                    ;;
                2)
                    print_info "Keeping existing training. Exiting."
                    exit 0
                    ;;
                3)
                    show_status
                    exit 0
                    ;;
                *)
                    print_error "Invalid choice"
                    exit 1
                    ;;
            esac
        else
            print_warning "Stale PID file found (process not running). Removing."
            rm "$PID_FILE"
        fi
    fi
}

create_directories() {
    mkdir -p "$LOG_DIR"
    mkdir -p "$CHECKPOINT_DIR"
    print_success "Directories created"
}

start_training() {
    print_info "Starting training in background..."

    # Start training with nohup
    # -u flag ensures unbuffered output
    cd "$PROJECT_ROOT"
    nohup python3 -u "$TRAINING_SCRIPT" > "$LOG_FILE" 2>&1 &

    # Save PID
    TRAINING_PID=$!
    echo "$TRAINING_PID" > "$PID_FILE"

    # Wait a moment to check if it started successfully
    sleep 2

    if ps -p "$TRAINING_PID" > /dev/null 2>&1; then
        print_success "Training started successfully!"
        echo ""
        echo -e "${GREEN}Process ID: $TRAINING_PID${NC}"
        echo -e "${GREEN}Log file: $LOG_FILE${NC}"
        echo ""
    else
        print_error "Training failed to start. Check $LOG_FILE for errors."
        tail -20 "$LOG_FILE"
        exit 1
    fi
}

show_monitoring_info() {
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}  Monitoring Commands${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
    echo ""
    echo "View live log output:"
    echo -e "  ${YELLOW}tail -f $LOG_FILE${NC}"
    echo ""
    echo "Monitor progress (refreshes every 5 seconds):"
    echo -e "  ${YELLOW}watch -n 5 python3 $SCRIPT_DIR/monitor_training.py${NC}"
    echo ""
    echo "Or use built-in watch mode:"
    echo -e "  ${YELLOW}python3 $SCRIPT_DIR/monitor_training.py --watch${NC}"
    echo ""
    echo "Check training status:"
    echo -e "  ${YELLOW}$SCRIPT_DIR/status.sh${NC}"
    echo ""
    echo "Stop training:"
    echo -e "  ${YELLOW}$SCRIPT_DIR/stop_training.sh${NC}"
    echo ""
    echo -e "${BLUE}=============================================================================${NC}"
}

show_status() {
    if [ ! -f "$PID_FILE" ]; then
        print_warning "No training process found"
        return
    fi

    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        print_success "Training is RUNNING (PID: $PID)"
        echo ""
        echo "Process info:"
        ps -p "$PID" -o pid,ppid,cmd,%cpu,%mem,etime
        echo ""
        if [ -f "$LOG_FILE" ]; then
            print_info "Last 10 lines of log:"
            tail -10 "$LOG_FILE"
        fi
    else
        print_error "Training is NOT running (stale PID file)"
        rm "$PID_FILE"
    fi
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    print_header
    echo ""

    print_info "Checking prerequisites..."
    check_python
    check_training_script
    check_existing_training

    echo ""
    print_info "Setting up environment..."
    create_directories

    echo ""
    start_training

    echo ""
    show_monitoring_info

    echo ""
    print_success "Setup complete! Training is running in the background."
    print_info "You can safely disconnect from SSH. Training will continue."
    echo ""
}

# Run main function
main
