#!/bin/bash

# Script to launch Jupyter notebooks for Simple RL

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Running setup first...${NC}"
    ./setup.sh
fi

# Activate virtual environment
source .venv/bin/activate

# Display available notebooks
echo -e "${GREEN}"
echo "======================================================"
echo "          Simple RL - Jupyter Notebooks              "
echo "======================================================"
echo -e "${NC}"
echo ""

if [ -d "notebooks" ]; then
    echo "Available notebooks:"
    echo ""
    for notebook in notebooks/*.ipynb; do
        if [ -f "$notebook" ]; then
            basename=$(basename "$notebook")
            echo -e "  ${BLUE}•${NC} $basename"
        fi
    done
    echo ""
fi

# Launch Jupyter
echo -e "${YELLOW}Launching Jupyter Notebook...${NC}"
echo "The server will open in your default browser."
echo "To stop the server, press Ctrl+C"
echo ""

cd notebooks 2>/dev/null || true
jupyter notebook