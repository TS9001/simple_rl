#!/bin/bash

# Unified Setup Script for Simple RL Project
# Uses uv for dependency management

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME=".venv"

# Functions
print_header() {
    echo -e "${GREEN}"
    echo "======================================================"
    echo "        Simple RL - Unified Environment Setup        "
    echo "======================================================"
    echo -e "${NC}"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if uv is installed
check_uv() {
    if command -v uv &> /dev/null; then
        print_success "uv is installed ($(uv --version))"
        return 0
    else
        print_warning "uv is not installed"
        return 1
    fi
}

# Install uv if not present
install_uv() {
    print_status "Installing uv..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install uv
        else
            curl -LsSf https://astral.sh/uv/install.sh | sh
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        print_error "Unsupported OS. Please install uv manually: https://github.com/astral-sh/uv"
        exit 1
    fi
    
    # Add to PATH if needed
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if command -v uv &> /dev/null; then
        print_success "uv installed successfully"
    else
        print_error "Failed to install uv"
        exit 1
    fi
}

# Setup Python environment with uv
setup_environment() {
    print_status "Setting up Python environment with uv..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment with uv
    if [ ! -d "$VENV_NAME" ]; then
        print_status "Creating virtual environment..."
        uv venv "$VENV_NAME" --python 3.11
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    print_success "Virtual environment activated"
}

# Install dependencies with uv
install_dependencies() {
    print_status "Installing project dependencies with uv..."
    
    cd "$PROJECT_ROOT"
    
    # Install main project in editable mode
    print_status "Installing simple_rl package..."
    uv pip install -e .
    print_success "simple_rl package installed"
    
    # Install PyTorch (CPU version for compatibility, users can upgrade to GPU version)
    print_status "Installing PyTorch..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    print_success "PyTorch installed"
    
    # Install ML dependencies
    print_status "Installing ML dependencies..."
    uv pip install transformers datasets accelerate tokenizers sentencepiece
    print_success "ML dependencies installed"
    
    # Install scientific computing
    print_status "Installing scientific computing packages..."
    uv pip install numpy pandas scipy matplotlib seaborn
    print_success "Scientific packages installed"
    
    # Install Jupyter
    print_status "Installing Jupyter..."
    uv pip install jupyter notebook ipykernel ipywidgets
    print_success "Jupyter installed"
    
    # Install configuration and logging
    print_status "Installing configuration tools..."
    uv pip install omegaconf hydra-core wandb rich
    print_success "Configuration tools installed"
    
    # Install development tools
    print_status "Installing development tools..."
    uv pip install pytest pytest-cov black isort flake8
    print_success "Development tools installed"
}

# Setup Jupyter kernel
setup_jupyter() {
    print_status "Configuring Jupyter kernel..."
    python -m ipykernel install --user --name=simple_rl --display-name="Simple RL"
    print_success "Jupyter kernel configured"
}

# Create project structure
create_project_structure() {
    print_status "Creating project directories..."
    
    cd "$PROJECT_ROOT"
    
    # Create necessary directories
    mkdir -p checkpoints
    mkdir -p models
    mkdir -p logs
    mkdir -p data
    mkdir -p notebooks
    mkdir -p how_to
    
    print_success "Project directories created"
}

# Check GPU availability
check_gpu() {
    print_status "Checking GPU availability..."
    
    python -c "
import torch
if torch.cuda.is_available():
    print('  GPU: ' + torch.cuda.get_device_name(0))
    print('  CUDA: ' + torch.version.cuda)
else:
    print('  No GPU detected - using CPU')
" || print_warning "Could not check GPU status"
}

# Main setup flow
main() {
    print_header
    
    # Check and install uv if needed
    if ! check_uv; then
        install_uv
    fi
    
    # Setup environment
    setup_environment
    
    # Install dependencies
    install_dependencies
    
    # Setup Jupyter
    setup_jupyter
    
    # Create project structure
    create_project_structure
    
    # Check GPU
    check_gpu
    
    # Success message
    echo ""
    print_success "Setup complete!"
    echo ""
    echo -e "${GREEN}Environment is ready!${NC}"
    echo ""
    echo "To activate the environment:"
    echo -e "  ${BLUE}source $VENV_NAME/bin/activate${NC}"
    echo ""
    echo "To launch Jupyter notebooks:"
    echo -e "  ${BLUE}./run_jupyter.sh${NC}"
    echo ""
    echo "To run tests:"
    echo -e "  ${BLUE}./run_tests.sh${NC}"
    echo ""
    echo "For more information, see:"
    echo -e "  ${BLUE}how_to/getting_started.md${NC}"
}

# Run main function
main