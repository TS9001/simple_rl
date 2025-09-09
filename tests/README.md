# Simple RL Tests

This directory contains comprehensive tests for the Simple RL framework, including model finetuning tests with real HuggingFace models.

## 📋 Test Overview

### Test Files
- `test_model_finetuning.py` - Unit tests with mock models
- `test_overfitting.py` - Integration tests with real DistilBERT model and tiny dataset
- `conftest.py` - Shared pytest fixtures (if present)

### Test Categories
- **Unit Tests**: Fast tests with mock components
- **Integration Tests**: Real model downloads and training
- **Overfitting Tests**: Verify model can learn on tiny datasets

## 🚀 Getting Started

### Prerequisites

1. **Install Simple RL package:**
   ```bash
   pip install -e .
   ```

2. **Install test dependencies:**
   ```bash
   pip install pytest pytest-cov
   ```

3. **Install HuggingFace dependencies for finetuning tests:**
   ```bash
   pip install transformers datasets torch
   ```

### Basic Test Commands

#### Run All Tests
```bash
pytest tests/
```

#### Run Specific Test File
```bash
# Fast unit tests only
pytest tests/test_model_finetuning.py -v

# Slow integration tests (downloads models)
pytest tests/test_overfitting.py -v
```

#### Run Specific Test Method
```bash
# Test overfitting on tiny dataset
pytest tests/test_overfitting.py::TestOverfitting::test_overfit_small_model_tiny_dataset -v -s

# Test frozen backbone finetuning
pytest tests/test_overfitting.py::TestOverfitting::test_overfit_with_frozen_backbone -v -s
```

## ⚡ Quick Test Options

### Fast Tests Only (No Model Downloads)
```bash
pytest tests/test_model_finetuning.py -v
```
- Uses mock models
- Completes in seconds
- Good for development

### Slow Tests (Real Models)
```bash
pytest tests/test_overfitting.py -v -s
```
- Downloads DistilBERT (~250MB)
- Takes 2-5 minutes
- Validates real finetuning

### Skip Slow Tests
```bash
pytest tests/ -m "not slow" -v
```

### Run Only Slow Tests
```bash
pytest tests/ -m "slow" -v -s
```

## 📊 Test Scenarios Explained

### 1. Overfitting Test (`test_overfit_small_model_tiny_dataset`)

**What it does:**
- Downloads DistilBERT-base-uncased (67M parameters)
- Creates tiny sentiment dataset (10 examples)
- Trains for 20 epochs to overfit
- Expects >80% accuracy on training data

**Expected output:**
```
Model has 66,955,010 parameters
Training on 10 examples
Final accuracy: 0.9000
Manual accuracy check: 90%
```

**Why it's useful:**
- Verifies the entire finetuning pipeline works
- Tests model can learn (sanity check)
- Validates save/load functionality

### 2. Frozen Backbone Test (`test_overfit_with_frozen_backbone`)

**What it does:**
- Freezes DistilBERT backbone parameters
- Only trains classification head (~1500 parameters)
- Uses higher learning rate (1e-3 vs 2e-5)
- Expects >60% accuracy

**Why it's useful:**
- Tests parameter freezing works correctly
- Faster training (fewer parameters)
- Common finetuning scenario

### 3. Save/Load Test (`test_save_and_load_finetuned_model`)

**What it does:**
- Trains model briefly
- Saves checkpoint to temporary file
- Loads checkpoint into new model
- Verifies predictions are identical

**Why it's useful:**
- Tests checkpoint format is correct
- Validates model state preservation
- Critical for production use

## 🔧 Test Configuration

### Environment Variables
```bash
# Skip slow tests by default
export PYTEST_SKIP_SLOW=1

# Run with more verbose output
export PYTEST_VERBOSE=1
```

### Custom pytest.ini
Create `pytest.ini` in project root:
```ini
[tool:pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
testpaths = tests
addopts = -v --tb=short
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Download Fails
```
ImportError: No module named 'transformers'
```
**Solution:**
```bash
pip install transformers datasets
```

#### 2. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Tests automatically use CPU, but if you have GPU issues:
```bash
export CUDA_VISIBLE_DEVICES=""
pytest tests/test_overfitting.py -v -s
```

#### 3. Slow Internet/Offline Testing
**Skip model download tests:**
```bash
pytest tests/ -m "not slow" -v
```

#### 4. Permission Errors (Checkpoint Saving)
**Solution:** Tests use temporary directories, but if issues persist:
```bash
chmod 755 tests/
mkdir -p checkpoints/
chmod 755 checkpoints/
```

### Debug Mode
```bash
# Run with full output and no capture
pytest tests/test_overfitting.py::TestOverfitting::test_overfit_small_model_tiny_dataset -v -s --tb=long

# Run with pdb debugger on failure
pytest tests/ --pdb -v
```

## 📈 Expected Test Results

### Successful Run Output:
```bash
$ pytest tests/test_overfitting.py -v -s

tests/test_overfitting.py::TestOverfitting::test_overfit_small_model_tiny_dataset 
Loading distilbert-base-uncased...
Model loaded. Hidden size: 768
Total parameters: 66,955,010
Training on 10 examples

Epoch 1/20 - Average Loss: 0.6234
Epoch 5/20 - Average Loss: 0.3456
Epoch 10/20 - Average Loss: 0.1234
Epoch 15/20 - Average Loss: 0.0456
Epoch 20/20 - Average Loss: 0.0123

Final accuracy: 0.9000
✅ 'This movie is amazing!' -> True: 1, Pred: 1
✅ 'I love this film so much.' -> True: 1, Pred: 1
...
Manual accuracy check: 90%
PASSED

tests/test_overfitting.py::TestOverfitting::test_overfit_with_frozen_backbone
Trainable parameters: 1,538
Frozen backbone - Accuracy: 0.7000
PASSED

=================== 2 passed in 120.45s ===================
```

## 🚦 CI/CD Integration

### GitHub Actions Example:
```yaml
name: Test Finetuning

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest transformers datasets
    
    - name: Run fast tests
      run: pytest tests/test_model_finetuning.py -v
    
    - name: Run slow tests (if not PR)
      if: github.event_name != 'pull_request'
      run: pytest tests/test_overfitting.py -v -s
```

## 💡 Tips for Development

1. **During development:** Run fast tests frequently
   ```bash
   pytest tests/test_model_finetuning.py -v
   ```

2. **Before committing:** Run slow tests once
   ```bash
   pytest tests/test_overfitting.py -v -s
   ```

3. **Debug specific issues:** Use individual test methods
   ```bash
   pytest tests/test_overfitting.py::TestOverfitting::test_overfit_small_model_tiny_dataset -v -s --tb=long
   ```

4. **Test with different models:** Modify model name in test files
   ```python
   model_name = "bert-base-uncased"  # Instead of distilbert
   ```