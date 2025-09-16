#!/bin/bash
# Setup Simple RL with uv package manager

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Function to check if uv is available
check_uv() {
    command -v uv &> /dev/null
}

# Install uv if needed and update PATH
if ! check_uv; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Verify installation
    if ! check_uv; then
        echo "❌ Failed to install uv. Please check the installation."
        exit 1
    fi
    
    echo "✅ uv installed successfully"
else
    echo "✅ uv is already available"
fi

echo "Setting up Python environment..."

# Create virtual environment
echo "Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.11
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies from pyproject.toml..."

# Install package with all dependencies
uv pip install -e .

# Install PyTorch separately with CPU version for compatibility
echo "Installing PyTorch (CPU version)..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Setup Jupyter kernel
echo "Setting up Jupyter kernel..."
python -m ipykernel install --user --name=simple_rl --display-name="Simple RL"

# Create necessary directories
echo "Creating project directories..."
mkdir -p checkpoints models logs data notebooks

# Verify installation
echo "Verifying installation..."
python -c "import simple_rl; print('✅ simple_rl package imported successfully')" 2>/dev/null || echo "⚠️  Warning: simple_rl package import failed"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the environment in a new terminal:"
echo "  source .venv/bin/activate"
echo ""
echo "Available scripts:"
echo "  📊 Run Jupyter: ./scripts/run_jupyter.sh"
echo "  🧪 Run tests: ./scripts/run_tests.sh grpo"
echo ""
echo "Your virtual environment is now active in this terminal session."