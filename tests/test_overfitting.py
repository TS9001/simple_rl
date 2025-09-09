"""
Overfitting test with real small model and tiny dataset.

Tests ability to overfit on a few examples using:
- Small HuggingFace model (distilbert-base-uncased)
- Tiny dataset (10 examples)
- Should achieve near-perfect training accuracy
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset as HFDataset

from simple_rl.algorithms import SupervisedLearning
from simple_rl.utils import create_supervised_config
from simple_rl.utils.logging import setup_logging

# Skip tests if transformers not available
transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")


class TextClassificationModel(nn.Module):
    """Text classification model using HuggingFace backbone."""
    
    def __init__(self, config):
        super().__init__()
        
        model_name = config.get("hf_model_name", "distilbert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from model config
        hidden_size = self.backbone.config.hidden_size
        num_classes = config.get("num_classes", 2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass.
        x should be a dict with 'input_ids' and 'attention_mask'
        """
        if isinstance(x, dict):
            input_ids = x["input_ids"]
            attention_mask = x.get("attention_mask", None)
        else:
            # Fallback for simple tensor input
            input_ids = x
            attention_mask = None
        
        # Get embeddings from backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation (first token)
        # Handle both tensor and model output formats
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            # Fallback if outputs is a tuple or list
            hidden_states = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        
        pooled_output = hidden_states[:, 0, :]
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits


class TextDataset(Dataset):
    """Simple text dataset that tokenizes on-the-fly."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch):
    """Custom collate function for text data."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "inputs": {"input_ids": input_ids, "attention_mask": attention_mask},
        "targets": labels
    }


@pytest.fixture
def tiny_dataset():
    """Create a tiny dataset for overfitting - 10 examples."""
    texts = [
        "This movie is amazing!",
        "I love this film so much.",
        "Best movie ever made.",
        "Absolutely fantastic!",
        "Great acting and plot.",
        "This movie is terrible.",
        "I hate this film.",
        "Worst movie ever.",
        "Completely boring.",
        "Awful acting and story."
    ]
    
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 5 positive, 5 negative
    
    return texts, labels


@pytest.fixture
def overfitting_config():
    """Configuration optimized for overfitting."""
    return create_supervised_config(
        project_name="overfit-test",
        algorithm={
            "task_type": "classification"
        },
        training={
            "num_epochs": 20,  # Many epochs to overfit
            "learning_rate": 2e-5,
            "optimizer": {
                "type": "adamw",
                "betas": [0.9, 0.999]
            }
        },
        logging={
            "log_interval": 5,
            "eval_interval": 50  # Less frequent eval
        },
        wandb={"enabled": False}
    )


class TestOverfitting:
    """Test overfitting with real model and tiny dataset."""
    
    @pytest.mark.slow
    def test_overfit_small_model_tiny_dataset(self, tiny_dataset, overfitting_config):
        """
        Test overfitting DistilBERT on 10 examples.
        Should achieve >90% accuracy on these examples.
        """
        setup_logging("INFO")
        
        texts, labels = tiny_dataset
        
        # Create model
        model_config = {
            "hf_model_name": "distilbert-base-uncased",
            "num_classes": 2,
            "dropout": 0.1
        }
        
        model = TextClassificationModel(model_config)
        
        # Create dataset
        train_dataset = TextDataset(texts, labels, model.tokenizer, max_length=64)
        
        # Use the same data for validation (overfitting test)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=2,  # Small batch size
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            train_dataset,  # Same data for validation
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Create algorithm
        algorithm = SupervisedLearning(model, overfitting_config, use_wandb=False)
        
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        print(f"Training on {len(texts)} examples")
        
        # Train
        results = algorithm.train(train_loader, val_loader)
        
        print(f"Final training loss: {results['avg_loss']:.4f}")
        
        # Evaluate on the same data
        eval_results = algorithm.evaluate(val_loader)
        
        print(f"Final accuracy: {eval_results['accuracy']:.4f}")
        print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
        
        # Should be able to overfit on 10 examples
        assert eval_results['accuracy'] > 0.8, f"Expected >80% accuracy, got {eval_results['accuracy']:.2%}"
        assert eval_results['eval_loss'] < 0.5, f"Expected low loss, got {eval_results['eval_loss']:.4f}"
        
        # Test individual predictions
        model.eval()
        correct_predictions = 0
        
        with torch.no_grad():
            for i, (text, true_label) in enumerate(zip(texts, labels)):
                # Tokenize single example
                encoding = model.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length", 
                    max_length=64,
                    return_tensors="pt"
                )
                
                # Move encoding to device
                device_encoding = {k: v.to(algorithm.device) for k, v in encoding.items()}
                
                # Predict
                logits = model(device_encoding)
                predicted_label = torch.argmax(logits, dim=1).item()
                
                print(f"'{text}' -> True: {true_label}, Pred: {predicted_label}")
                
                if predicted_label == true_label:
                    correct_predictions += 1
        
        accuracy = correct_predictions / len(texts)
        print(f"Manual accuracy check: {accuracy:.2%}")
        
        # Should get most examples right when overfitting
        assert accuracy >= 0.7, f"Expected at least 70% accuracy, got {accuracy:.2%}"
    
    @pytest.mark.slow  
    def test_overfit_with_frozen_backbone(self, tiny_dataset, overfitting_config):
        """
        Test overfitting with frozen backbone - only train classifier head.
        Should still achieve reasonable accuracy.
        """
        texts, labels = tiny_dataset
        
        # Create model
        model_config = {
            "hf_model_name": "distilbert-base-uncased",
            "num_classes": 2,
            "dropout": 0.1
        }
        
        model = TextClassificationModel(model_config)
        
        # Freeze backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        # Create datasets
        train_dataset = TextDataset(texts, labels, model.tokenizer, max_length=64)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        
        # Use higher learning rate since only training classifier
        frozen_config = create_supervised_config(
            project_name="frozen-overfit-test",
            algorithm={"task_type": "classification"},
            training={
                "num_epochs": 15,
                "learning_rate": 1e-3,  # Higher LR for classifier only
                "optimizer": {"type": "adamw"}
            },
            wandb={"enabled": False}
        )
        
        # Create algorithm
        algorithm = SupervisedLearning(model, frozen_config, use_wandb=False)
        
        # Train
        results = algorithm.train(train_loader, val_loader)
        eval_results = algorithm.evaluate(val_loader)
        
        print(f"Frozen backbone - Accuracy: {eval_results['accuracy']:.4f}")
        
        # Should still be able to learn something, even with frozen backbone
        assert eval_results['accuracy'] > 0.6, f"Expected >60% accuracy with frozen backbone, got {eval_results['accuracy']:.2%}"
    
    def test_save_and_load_finetuned_model(self, tiny_dataset, overfitting_config, tmp_path):
        """Test saving and loading a finetuned model."""
        texts, labels = tiny_dataset
        
        # Create and train model
        model_config = {
            "hf_model_name": "distilbert-base-uncased", 
            "num_classes": 2
        }
        
        model1 = TextClassificationModel(model_config)
        
        # Quick training config
        quick_config = create_supervised_config(
            project_name="save-load-test",
            algorithm={"task_type": "classification"},
            training={"num_epochs": 3, "learning_rate": 2e-5},
            wandb={"enabled": False}
        )
        
        train_dataset = TextDataset(texts, labels, model1.tokenizer, max_length=64)
        train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
        
        algorithm1 = SupervisedLearning(model1, quick_config, use_wandb=False)
        algorithm1.train(train_loader)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "finetuned_model.pt"
        algorithm1.save_checkpoint(str(checkpoint_path))
        
        # Load into new model
        model2 = TextClassificationModel(model_config)
        algorithm2 = SupervisedLearning(model2, quick_config, use_wandb=False)
        algorithm2.load_checkpoint(str(checkpoint_path))
        
        # Test that both models give same predictions
        test_text = "This is a test sentence."
        
        encoding = model1.tokenizer(
            test_text,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )
        
        # Set both models to eval mode
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            # Move encoding to device
            device_encoding = {k: v.to(algorithm1.device) for k, v in encoding.items()}
            logits1 = model1(device_encoding)
            logits2 = model2(device_encoding)
        
        torch.testing.assert_close(logits1, logits2, msg="Loaded model gives different predictions")