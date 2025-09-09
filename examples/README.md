# Simple RL Examples

This directory contains practical examples demonstrating how to use Simple RL for various tasks.

## 📁 Example Files

### Model Finetuning
- `finetune_example.py` - Complete DistilBERT finetuning on tiny sentiment dataset
- `finetune_config.yaml` - Configuration examples for different finetuning scenarios

### Configuration Examples  
- `config_based_training.py` - Using external YAML configs with command line overrides
- `train_with_custom_model.py` - Training custom PyTorch models with Simple RL
- `custom_optimizer.yaml` - Examples of using custom optimizers (Muon, Lion, etc.)
- `custom_optimizer_example.py` - Register and use custom optimizers

### Testing
- `basic_usage.py` - Simple example showing basic package functionality

## 🚀 Running Examples

### Prerequisites

1. **Install Simple RL:**
   ```bash
   pip install -e .
   ```

2. **Install optional dependencies for specific examples:**
   ```bash
   # For finetuning examples
   pip install transformers datasets
   
   # For custom optimizers (optional)
   pip install lion-pytorch  # For Lion optimizer
   # pip install muon        # For Muon optimizer
   ```

## 📊 Finetuning Example (Recommended Start)

### Quick Start
```bash
python examples/finetune_example.py
```

**What this does:**
- Downloads DistilBERT-base-uncased (~250MB, one-time)
- Creates tiny sentiment dataset (10 examples)
- Finetunes for 15 epochs to overfit
- Shows before/after predictions
- Saves and loads the finetuned model

**Expected output:**
```
🚀 Starting DistilBERT finetuning on tiny sentiment dataset
Loading distilbert-base-uncased...
📚 Dataset: 10 examples  
🏗️ Model: 66,955,010 parameters

🎲 Initial predictions (before training):
❌ NEG (0.523): 'This movie is absolutely amazing! Best film ever.'
❌ POS (0.487): 'This movie is terrible and boring.'
🎯 Accuracy: 40% (4/10)

🏋️ Starting training for 15 epochs...
Epoch 1/15 - Average Loss: 0.6234
Epoch 5/15 - Average Loss: 0.2145  
Epoch 10/15 - Average Loss: 0.0567
Epoch 15/15 - Average Loss: 0.0123

🧪 Final predictions (after training):
✅ POS (0.967): 'This movie is absolutely amazing! Best film ever.'
✅ NEG (0.943): 'This movie is terrible and boring.'
🎯 Accuracy: 100% (10/10)

📈 RESULTS SUMMARY:
   Initial accuracy: 40%
   Final accuracy: 100%
   Improvement: +60%

🎉 SUCCESS: Model successfully overfitted on tiny dataset!
```

**Time to complete:** ~2-3 minutes (including model download)

## 🔧 Configuration Examples

### Using External Config Files
```bash
# Basic usage with default config
python examples/config_based_training.py

# Use specific config file  
python examples/config_based_training.py --config simple_rl/config/algorithms/supervised_learning.yaml

# Override parameters
python examples/config_based_training.py --learning-rate 1e-3 --num-epochs 50

# Different task type
python examples/config_based_training.py --task-type classification --enable-wandb
```

### Custom Optimizer Examples
```bash
# Register optimizer at runtime
python examples/custom_optimizer_example.py

# Use optimizer from external package
python examples/config_based_training.py --config examples/custom_optimizer.yaml
```

## 📋 Example Descriptions

### 1. `finetune_example.py` ⭐ **Start Here**
**Purpose:** Complete finetuning walkthrough with real model  
**Model:** DistilBERT-base-uncased (67M parameters)  
**Dataset:** 10 sentiment examples  
**Goal:** Demonstrate overfitting for validation  
**Time:** 2-3 minutes  

**Key features:**
- Real HuggingFace model integration
- Before/after prediction comparison  
- Checkpoint saving and loading
- Detailed progress logging

### 2. `config_based_training.py`
**Purpose:** Show external configuration management  
**Model:** Simple MLP (configurable)  
**Dataset:** Synthetic data  
**Goal:** Demonstrate config flexibility  
**Time:** <1 minute

**Key features:**
- YAML configuration files
- Command line parameter overrides  
- Algorithm comparison
- Config validation and merging

### 3. `train_with_custom_model.py`  
**Purpose:** Train any PyTorch model with Simple RL  
**Model:** Custom MLP  
**Dataset:** Synthetic regression data  
**Goal:** Show algorithm/model decoupling  
**Time:** <1 minute

**Key features:**
- Custom model architecture
- Optimizer configuration from config
- Learning rate scheduling
- Early stopping

### 4. `custom_optimizer_example.py`
**Purpose:** Add custom optimizers (Muon, Lion, etc.)  
**Model:** Simple test model  
**Dataset:** Synthetic data  
**Goal:** Show optimizer extensibility  
**Time:** <1 minute

**Key features:**
- Runtime optimizer registration
- Import-path based optimizers
- Parameter filtering
- Performance comparison

### 5. `basic_usage.py`
**Purpose:** Package functionality overview  
**Model:** Mock components  
**Dataset:** No real data  
**Goal:** Verify installation  
**Time:** <30 seconds

## 🎯 Choosing the Right Example

### For Learning Simple RL:
1. **Start with:** `basic_usage.py` (verify installation)
2. **Then try:** `finetune_example.py` (see real results)  
3. **Explore:** `config_based_training.py` (understand configs)

### For Development:
1. **Model Integration:** `train_with_custom_model.py`
2. **Config Management:** `config_based_training.py`  
3. **Custom Components:** `custom_optimizer_example.py`

### For Research:
1. **Finetuning:** `finetune_example.py` + modify for your data
2. **Hyperparameter Search:** `config_based_training.py` + grid search
3. **New Optimizers:** `custom_optimizer_example.py` + your optimizer

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Download Fails
```
requests.exceptions.ConnectionError
```
**Solutions:**
```bash
# Use HF_HUB_OFFLINE if models already cached
export HF_HUB_OFFLINE=1

# Or manually download
python -c "from transformers import AutoModel; AutoModel.from_pretrained('distilbert-base-uncased')"
```

#### 2. Memory Issues
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Examples automatically use CPU
- Reduce batch size in configs
- Use smaller model: `"prajjwal1/bert-tiny"`

#### 3. Import Errors
```
ImportError: No module named 'transformers'
```
**Solutions:**
```bash
pip install transformers datasets torch
```

#### 4. Slow Training
**Normal behavior:**
- First run: Downloads model (~2-3 min)
- Subsequent runs: Uses cached model (<1 min)
- Overfitting 10 examples: Should be very fast

### Performance Tips

1. **Faster models for testing:**
   ```python
   model_name = "prajjwal1/bert-tiny"  # 4M parameters
   model_name = "distilbert-base-uncased"  # 67M parameters  
   model_name = "bert-base-uncased"  # 110M parameters
   ```

2. **Reduce dataset size:**
   ```python
   # In finetune_example.py, modify TinySentimentDataset
   self.examples = self.examples[:5]  # Use only 5 examples
   ```

3. **Fewer epochs:**
   ```python
   training={"num_epochs": 5}  # Instead of 15
   ```

## 🚦 CI/CD Testing

### Quick Validation (30 seconds):
```bash
python examples/basic_usage.py
python examples/train_with_custom_model.py  
```

### Full Validation (3 minutes):
```bash
python examples/finetune_example.py
```

### Config Validation:
```bash
python examples/config_based_training.py --num-epochs 3
```

This should give you everything you need to get started with the examples and understand what each one demonstrates!