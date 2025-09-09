# Simple RL

A Python package for implementing simple, educational versions of reinforcement learning algorithms with HuggingFace model integration, dataset support, and Weights & Biases tracking.

## Features

- 🤖 **Simple RL Algorithms**: Clean, educational implementations of common RL algorithms
- 🤗 **HuggingFace Integration**: Load pretrained models and datasets from HuggingFace Hub
- 📊 **Experiment Tracking**: Built-in Weights & Biases integration for experiment tracking
- 🔧 **Flexible Configuration**: YAML-based configuration with easy overrides
- 🐍 **PyTorch Backend**: Built on PyTorch for GPU acceleration and modern ML workflows

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd simple_rl

# Install in development mode
pip install -e ".[dev]"

# Or use the setup script
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

### Basic Usage

```python
from simple_rl.utils.config import get_default_config
from simple_rl.algorithms.base import BaseAlgorithm

# Load configuration
config = get_default_config()

# Initialize algorithm
algorithm = BaseAlgorithm(config)

# Train (placeholder - implement specific algorithms)
# results = algorithm.train(num_episodes=1000)
```

### Command Line Training

```bash
# Train with default configuration
simple-rl-train

# Train with custom config
simple-rl-train --config my_config.yaml

# Override specific parameters
simple-rl-train --override training.num_episodes=2000 model.hidden_dim=512
```

## Project Structure

```
simple_rl/
├── simple_rl/
│   ├── algorithms/       # RL algorithm implementations
│   │   ├── base.py      # Base algorithm class
│   │   └── __init__.py
│   ├── models/          # Model definitions
│   │   ├── base.py      # Base model with HF integration
│   │   └── __init__.py
│   ├── utils/           # Utilities
│   │   ├── config.py    # Configuration management
│   │   ├── logging.py   # Logging utilities
│   │   ├── data.py      # Data loading utilities
│   │   └── __init__.py
│   ├── config/          # Configuration files
│   │   └── default.yaml # Default configuration
│   └── scripts/         # Training scripts
│       └── train.py     # Main training script
├── examples/            # Usage examples
├── tests/              # Unit tests
├── scripts/            # Setup scripts
└── requirements.txt    # Dependencies
```

## Configuration

The package uses YAML configuration files with OmegaConf. See `simple_rl/config/default.yaml` for all available options.

Key configuration sections:
- `model`: Model architecture and HuggingFace integration
- `training`: Training hyperparameters and algorithm settings  
- `data`: Dataset loading and preprocessing
- `wandb`: Experiment tracking configuration
- `environment`: Gym environment settings

## Development

### Code Quality

```bash
# Format code
black simple_rl/ tests/ examples/

# Sort imports
isort simple_rl/ tests/ examples/

# Lint code
flake8 simple_rl/ tests/ examples/

# Run all checks
black . && isort . && flake8 .
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=simple_rl
```

## Extending Simple RL

### Adding New Algorithms

1. Create a new file in `simple_rl/algorithms/`
2. Inherit from `BaseAlgorithm`
3. Implement `train()` and `evaluate()` methods
4. Register in `algorithms/__init__.py`

### Adding New Models

1. Create a new file in `simple_rl/models/`  
2. Inherit from `BaseModel`
3. Implement `forward()` method
4. Register in `models/__init__.py`

## Dependencies

Core dependencies:
- PyTorch 2.0+
- HuggingFace Transformers & Datasets
- Weights & Biases
- OpenAI Gym
- OmegaConf

Development dependencies:
- pytest, black, flake8, isort

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run code quality checks
6. Submit a pull request