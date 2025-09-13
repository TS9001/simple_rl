#!/bin/bash
# Launch Jupyter for GRPO training

set -e

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo "Setting up environment first..."
    ./scripts/setup.sh
fi

source .venv/bin/activate

echo "Starting Jupyter Notebook..."
echo "Available notebooks:"
ls -la notebooks/*.ipynb 2>/dev/null | awk '{print "  • " $NF}'
echo ""

cd notebooks 2>/dev/null || true
jupyter notebook