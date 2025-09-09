#!/usr/bin/env python3
"""
Practical example of finetuning a small model on a tiny dataset.

This example shows:
1. Loading a small HuggingFace model (DistilBERT)
2. Creating a tiny sentiment dataset (10 examples)
3. Overfitting the model to achieve near-perfect accuracy
4. Saving and loading the finetuned model
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_rl.algorithms import SupervisedLearning
from simple_rl.models.base import BaseModel
from simple_rl.utils import create_supervised_config
from simple_rl.utils.logging import setup_logging, get_logger


class SentimentModel(BaseModel):
    """Sentiment classification model using DistilBERT."""
    
    def __init__(self, config):
        super().__init__(config)
        
        model_name = "distilbert-base-uncased"
        print(f"Loading {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Classification head
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)  # Binary sentiment
        )
        
        print(f"Model loaded. Hidden size: {hidden_size}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """Forward pass with input dict containing input_ids and attention_mask."""
        outputs = self.backbone(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"]
        )
        
        # Use [CLS] token (first token) for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


class TinySentimentDataset(Dataset):
    """Tiny sentiment dataset for overfitting."""
    
    def __init__(self, tokenizer, max_length=128):
        # 10 examples: 5 positive, 5 negative
        self.examples = [
            ("This movie is absolutely amazing! Best film ever.", 1),
            ("I love this movie so much, incredible acting.", 1), 
            ("Fantastic storyline and great characters.", 1),
            ("Wonderful cinematography and perfect ending.", 1),
            ("Brilliant performance by all actors.", 1),
            ("This movie is terrible and boring.", 0),
            ("Worst film I've ever seen, complete waste of time.", 0),
            ("Awful acting and confusing plot.", 0),
            ("Boring story with no character development.", 0),
            ("Completely disappointing and poorly made.", 0)
        ]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text, label = self.examples[idx]
        
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
            "label": torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch):
    """Custom collate function."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    
    return {
        "inputs": {"input_ids": input_ids, "attention_mask": attention_mask},
        "targets": labels
    }


def test_predictions(model, dataset, logger):
    """Test model predictions on all examples."""
    model.eval()
    correct = 0
    
    logger.info("\n📊 Testing predictions on all examples:")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            text, true_label = dataset.examples[i]
            
            # Add batch dimension
            inputs = {
                "input_ids": item["input_ids"].unsqueeze(0),
                "attention_mask": item["attention_mask"].unsqueeze(0)
            }
            
            # Predict
            logits = model(inputs)
            predicted_label = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
            
            status = "✅" if predicted_label == true_label else "❌"
            sentiment = "POS" if predicted_label == 1 else "NEG"
            
            logger.info(f"{status} {sentiment} ({confidence:.3f}): '{text[:50]}...'")
            
            if predicted_label == true_label:
                correct += 1
    
    accuracy = correct / len(dataset)
    logger.info(f"\n🎯 Accuracy: {accuracy:.2%} ({correct}/{len(dataset)})")
    return accuracy


def main():
    """Main finetuning example."""
    setup_logging("INFO")
    logger = get_logger("finetune_example")
    
    logger.info("🚀 Starting DistilBERT finetuning on tiny sentiment dataset")
    
    # Create model
    model_config = {"num_classes": 2}
    model = SentimentModel(model_config)
    
    # Create tiny dataset
    dataset = TinySentimentDataset(model.tokenizer, max_length=64)
    
    # Use same data for train and validation (overfitting test)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    logger.info(f"📚 Dataset: {len(dataset)} examples")
    logger.info(f"🏗️ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test initial predictions (should be random)
    logger.info("\n🎲 Initial predictions (before training):")
    initial_accuracy = test_predictions(model, dataset, logger)
    
    # Configuration for overfitting
    config = create_supervised_config(
        project_name="distilbert-sentiment-overfit",
        run_name="tiny-dataset-overfit",
        algorithm={"task_type": "classification"},
        training={
            "num_epochs": 15,
            "learning_rate": 2e-5,
            "optimizer": {
                "type": "adamw",
                "betas": [0.9, 0.999]
            }
        },
        logging={
            "log_interval": 5,
            "eval_interval": 20
        },
        wandb={"enabled": False}
    )
    
    # Create algorithm and train
    algorithm = SupervisedLearning(model, config, use_wandb=False)
    
    logger.info(f"\n🏋️ Starting training for {config.training.num_epochs} epochs...")
    
    # Train model
    results = algorithm.train(train_loader, val_loader)
    
    logger.info(f"\n✅ Training completed!")
    logger.info(f"📉 Final training loss: {results['avg_loss']:.4f}")
    logger.info(f"🔄 Total epochs: {results['total_epochs']}")
    
    # Final evaluation
    eval_results = algorithm.evaluate(val_loader)
    logger.info(f"📊 Final validation accuracy: {eval_results['accuracy']:.2%}")
    logger.info(f"📉 Final validation loss: {eval_results['eval_loss']:.4f}")
    
    # Test individual predictions
    logger.info("\n🧪 Final predictions (after training):")
    final_accuracy = test_predictions(model, dataset, logger)
    
    # Summary
    logger.info(f"\n📈 RESULTS SUMMARY:")
    logger.info(f"   Initial accuracy: {initial_accuracy:.2%}")
    logger.info(f"   Final accuracy: {final_accuracy:.2%}")
    logger.info(f"   Improvement: {final_accuracy - initial_accuracy:+.1%}")
    
    # Save model
    checkpoint_path = "checkpoints/distilbert_sentiment_overfit.pt"
    algorithm.save_checkpoint(checkpoint_path)
    logger.info(f"💾 Model saved to {checkpoint_path}")
    
    # Test loading
    logger.info("\n🔄 Testing model loading...")
    new_model = SentimentModel(model_config)
    new_algorithm = SupervisedLearning(new_model, config, use_wandb=False)
    new_algorithm.load_checkpoint(checkpoint_path)
    
    # Verify loaded model works
    test_accuracy = test_predictions(new_model, dataset, logger)
    logger.info(f"✅ Loaded model accuracy: {test_accuracy:.2%}")
    
    if final_accuracy > 0.8:
        logger.info("🎉 SUCCESS: Model successfully overfitted on tiny dataset!")
    else:
        logger.warning("⚠️ Model may need more training to overfit completely.")
    
    logger.info("\n🏁 Finetuning example completed!")


if __name__ == "__main__":
    main()