#!/bin/bash
# ==============================================================================
# Log Viewer
# ==============================================================================
# Interactive log viewer with multiple viewing options
#
# Usage:
#   ./scripts/view_logs.sh [option]
#
# Options:
#   --tail, -t       : View last 50 lines
#   --follow, -f     : Follow log in real-time
#   --errors, -e     : Show only errors/warnings
#   --full           : View full log
#   --progress, -p   : View progress JSON
# ==============================================================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/training.log"
PROGRESS_FILE="$PROJECT_ROOT/logs/progress.json"

# Colors
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ==============================================================================
# Functions
# ==============================================================================

check_log_exists() {
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${RED}✗ Log file not found: $LOG_FILE${NC}"
        echo "Training may not have started yet."
        exit 1
    fi
}

view_tail() {
    check_log_exists
    echo -e "${BLUE}Last 50 lines of training log:${NC}"
    echo "============================================================================="
    tail -50 "$LOG_FILE"
}

view_follow() {
    check_log_exists
    echo -e "${BLUE}Following training log (Ctrl+C to stop):${NC}"
    echo "============================================================================="
    tail -f "$LOG_FILE"
}

view_errors() {
    check_log_exists
    echo -e "${BLUE}Errors and warnings:${NC}"
    echo "============================================================================="
    grep -i -E "(error|warning|exception|failed|traceback)" "$LOG_FILE" | tail -50
    echo ""
    echo "Showing last 50 error/warning lines"
}

view_full() {
    check_log_exists
    echo -e "${BLUE}Opening full log in less (q to quit):${NC}"
    sleep 1
    less +G "$LOG_FILE"
}

view_progress() {
    if [ ! -f "$PROGRESS_FILE" ]; then
        echo -e "${RED}✗ Progress file not found: $PROGRESS_FILE${NC}"
        exit 1
    fi

    echo -e "${BLUE}Training Progress:${NC}"
    echo "============================================================================="

    if command -v python3 &> /dev/null; then
        python3 "$SCRIPT_DIR/monitor_training.py"
    else
        cat "$PROGRESS_FILE" | python3 -m json.tool
    fi
}

view_metrics() {
    check_log_exists
    echo -e "${BLUE}Episode metrics:${NC}"
    echo "============================================================================="
    grep -E "Episode [0-9]+" "$LOG_FILE" | tail -20
}

show_menu() {
    echo "============================================================================="
    echo "  Log Viewer"
    echo "============================================================================="
    echo ""
    echo "Choose an option:"
    echo "  1) View last 50 lines"
    echo "  2) Follow log in real-time"
    echo "  3) Show errors/warnings only"
    echo "  4) View full log"
    echo "  5) View progress JSON"
    echo "  6) View episode metrics"
    echo "  q) Quit"
    echo ""
    read -p "Enter choice: " choice

    case $choice in
        1) view_tail ;;
        2) view_follow ;;
        3) view_errors ;;
        4) view_full ;;
        5) view_progress ;;
        6) view_metrics ;;
        q|Q) exit 0 ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
}

# ==============================================================================
# Main
# ==============================================================================

case "$1" in
    --tail|-t)
        view_tail
        ;;
    --follow|-f)
        view_follow
        ;;
    --errors|-e)
        view_errors
        ;;
    --full)
        view_full
        ;;
    --progress|-p)
        view_progress
        ;;
    --metrics|-m)
        view_metrics
        ;;
    "")
        show_menu
        ;;
    *)
        echo "Unknown option: $1"
        echo ""
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  --tail, -t       : View last 50 lines"
        echo "  --follow, -f     : Follow log in real-time"
        echo "  --errors, -e     : Show only errors/warnings"
        echo "  --full           : View full log"
        echo "  --progress, -p   : View progress JSON"
        echo "  --metrics, -m    : View episode metrics"
        exit 1
        ;;
esac
