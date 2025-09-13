#!/bin/bash
# Setup Simple RL with uv package manager

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Install uv if needed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Setting up Python environment..."

# Create virtual environment
if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.11
fi

# Activate and install dependencies
source .venv/bin/activate

echo "Installing dependencies from pyproject.toml..."

# Install package with all dependencies
uv pip install -e .

# Install PyTorch separately with CPU version for compatibility
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Setup Jupyter kernel
python -m ipykernel install --user --name=simple_rl --display-name="Simple RL"

# Create directories
mkdir -p checkpoints models logs data notebooks

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate: source .venv/bin/activate"
echo "To run GRPO: ./scripts/run_jupyter.sh"
echo "To test: ./scripts/run_tests.sh grpo"