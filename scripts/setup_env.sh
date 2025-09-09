#!/bin/bash
# Setup script for Simple RL development environment

set -e

echo "Setting up Simple RL development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "Installing Simple RL in development mode..."
pip install -e ".[dev]"

echo "Environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"