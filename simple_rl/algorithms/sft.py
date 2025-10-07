"""
Supervised Fine-Tuning (SFT) for language models.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
import time
import numpy as np

from .base import BaseAlgorithm


class SFTDataset(Dataset):
    """
    PyTorch Dataset for supervised fine-tuning of language models.
    Creates input_ids and labels where only the completion tokens contribute to loss.
    """

    def __init__(
        self,
        prompts: List[str],
        completions: List[str],
        tokenizer,
        max_length: int = 1024,
    ):
        """
        Initialize SFT dataset.

        Args:
            prompts: List of prompt strings
            completions: List of completion strings (targets)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.prompts = prompts
        self.completions = completions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        completion = self.completions[idx]

        # Full text = prompt + completion
        full_text = prompt + "\n" + completion

        # Tokenize full text
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize just the prompt to find where completion starts
        prompt_encoding = self.tokenizer(
            prompt + "\n",
            truncation=True,
            max_length=self.max_length,
        )
        prompt_length = len(prompt_encoding["input_ids"])

        # Create labels: -100 for prompt tokens (ignored in loss), actual token ids for completion
        labels = encoding["input_ids"].clone()
        labels[0, :prompt_length] = -100  # Mask prompt tokens
        labels[0, encoding["attention_mask"][0] == 0] = -100  # Mask padding tokens

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


class SFT(BaseAlgorithm):
    """
    Supervised Fine-Tuning (SFT) algorithm for language models.

    Trains language models using standard cross-entropy loss on prompted completions.
    Only the completion tokens contribute to the loss (prompt tokens are masked).
    """

    def __init__(
        self,
        config: dict,
        tokenizer: Optional[Any] = None,
        model: Optional[nn.Module] = None,
        use_wandb: bool = False,
    ):
        """
        Initialize SFT algorithm.

        Args:
            config: Configuration dictionary
            tokenizer: HuggingFace tokenizer (optional, will be loaded from config if not provided)
            model: HuggingFace model (optional, will be loaded from config if not provided)
            use_wandb: Whether to use Weights & Biases logging
        """
        self.config = config
        self.use_wandb = use_wandb

        # Setup device
        device_config = config.get("model", {}).get("device", "auto")
        if device_config == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_config)

        # Load tokenizer
        if tokenizer is None:
            model_name = config["model"]["model_name"]
            print(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        # Load model
        if model is None:
            model_name = config["model"]["model_name"]
            print(f"Loading model: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
        else:
            self.model = model

        self.model = self.model.to(self.device)

        # Training configuration
        training_config = config.get("training", {})
        self.batch_size = training_config.get("batch_size", 4)
        self.learning_rate = training_config.get("learning_rate", 5e-5)
        self.num_epochs = training_config.get("num_epochs", 3)
        self.gradient_accumulation_steps = training_config.get(
            "gradient_accumulation_steps", 4
        )
        self.max_grad_norm = training_config.get("max_grad_norm", 1.0)
        self.warmup_steps = training_config.get("warmup_steps", 100)

        # Model configuration
        model_config = config.get("model", {})
        self.max_length = model_config.get("max_length", 1024)

        # Logging configuration
        logging_config = config.get("logging", {})
        self.log_interval = logging_config.get("log_interval", 50)
        self.save_interval = logging_config.get("save_interval", 500)

        # Validation configuration
        validation_config = config.get("validation", {})
        self.validation_enabled = validation_config.get("enabled", True)
        self.validation_interval = validation_config.get("interval", 200)
        self.validation_num_samples = validation_config.get("num_samples", 20)

        # Initialize optimizer (will be set up in train())
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.global_step = 0

        # Initialize wandb if requested
        if use_wandb:
            import wandb

            wandb_config = config.get("wandb", {})
            wandb.init(
                project=wandb_config.get("project", "simple-rl-sft"),
                config=config,
                name=wandb_config.get("run_name", None),
            )

        print(f"✓ SFT initialized with model: {config['model']['model_name']}")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Device: {self.device}")

    def train(
        self,
        train_data: Dict[str, List[str]],
        val_data: Optional[Dict[str, List[str]]] = None,
        num_episodes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Train the model using supervised fine-tuning.

        Args:
            train_data: Dict with 'prompts' and 'completions' lists
            val_data: Optional validation data with same format
            num_episodes: Number of epochs to train (overrides config)

        Returns:
            Dictionary of training metrics
        """
        # Create dataset and dataloader
        train_dataset = SFTDataset(
            prompts=train_data["prompts"],
            completions=train_data["completions"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Calculate total training steps
        num_epochs = num_episodes or self.num_epochs
        num_training_steps = (
            len(train_dataloader) * num_epochs // self.gradient_accumulation_steps
        )

        # Initialize optimizer if not already done
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
            )

        # Initialize scheduler if not already done
        if self.scheduler is None:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=num_training_steps,
            )

        print(f"Starting SFT training for {num_epochs} epochs...")
        print(f"Batch size: {self.batch_size}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(
            f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}"
        )
        print("=" * 50)

        # Training metrics storage
        training_metrics = {
            "step": [],
            "loss": [],
            "learning_rate": [],
            "epoch": [],
            "tokens_per_second": [],
        }

        validation_metrics = {
            "step": [],
            "loss": [],
        }

        # Create checkpoint directory
        checkpoint_dir = Path("checkpoints/sft")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        self.model.train()
        training_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_start_time = time.time()

            progress_bar = tqdm(
                train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"
            )

            for batch_idx, batch in enumerate(progress_bar):
                batch_start_time = time.time()

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item()

                # Update weights after accumulation steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Calculate tokens per second
                    batch_time = time.time() - batch_start_time
                    num_tokens = attention_mask.sum().item()
                    tokens_per_sec = num_tokens / batch_time if batch_time > 0 else 0

                    # Store metrics
                    current_lr = self.scheduler.get_last_lr()[0]
                    training_metrics["step"].append(self.global_step)
                    training_metrics["loss"].append(
                        loss.item() * self.gradient_accumulation_steps
                    )
                    training_metrics["learning_rate"].append(current_lr)
                    training_metrics["epoch"].append(epoch)
                    training_metrics["tokens_per_second"].append(tokens_per_sec)

                    # Update progress bar
                    progress_bar.set_postfix(
                        {
                            "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                            "lr": f"{current_lr:.2e}",
                            "tok/s": f"{tokens_per_sec:.0f}",
                        }
                    )

                    # Logging
                    if self.global_step % self.log_interval == 0:
                        print(
                            f"\nStep {self.global_step} | "
                            f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Tokens/s: {tokens_per_sec:.0f}"
                        )

                        if self.use_wandb:
                            import wandb

                            wandb.log(
                                {
                                    "train/loss": loss.item()
                                    * self.gradient_accumulation_steps,
                                    "train/learning_rate": current_lr,
                                    "train/tokens_per_second": tokens_per_sec,
                                    "train/step": self.global_step,
                                    "train/epoch": epoch,
                                }
                            )

                    # Validation
                    if (
                        self.validation_enabled
                        and val_data is not None
                        and self.global_step % self.validation_interval == 0
                    ):
                        print(f"\n{'='*60}")
                        print(f"VALIDATION AT STEP {self.global_step}")
                        print(f"{'='*60}")

                        val_metrics = self._validate(val_data)

                        validation_metrics["step"].append(self.global_step)
                        validation_metrics["loss"].append(val_metrics["loss"])

                        print(f"Validation Loss: {val_metrics['loss']:.4f}")

                        if self.use_wandb:
                            import wandb

                            wandb.log(
                                {
                                    "val/loss": val_metrics["loss"],
                                    "val/step": self.global_step,
                                }
                            )

                        self.model.train()  # Back to training mode

                    # Save checkpoint
                    if self.global_step % self.save_interval == 0:
                        checkpoint_path = (
                            checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
                        )
                        self.save_checkpoint(str(checkpoint_path))
                        print(f"  → Saved checkpoint to {checkpoint_path}")

            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(
                f"\nEpoch {epoch+1} completed in {epoch_time:.2f}s | Avg Loss: {avg_epoch_loss:.4f}"
            )

        # Training complete
        total_training_time = time.time() - training_start_time
        print("=" * 50)
        print("Training complete!")
        print(
            f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)"
        )
        print(f"Total steps: {self.global_step}")

        return {
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "total_time": total_training_time,
            "final_loss": training_metrics["loss"][-1] if training_metrics["loss"] else None,
        }

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Dictionary with 'prompts' and 'completions'

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        # Create mini-dataset for this batch
        dataset = SFTDataset(
            prompts=batch["prompts"],
            completions=batch["completions"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=len(batch["prompts"]))

        # Get the batch
        data_batch = next(iter(dataloader))

        # Move to device
        input_ids = data_batch["input_ids"].to(self.device)
        attention_mask = data_batch["attention_mask"].to(self.device)
        labels = data_batch["labels"].to(self.device)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss

        # Backward pass
        if self.optimizer is None:
            # Initialize optimizer if needed
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
            )

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {"loss": loss.item()}

    def evaluate(
        self,
        test_data: Dict[str, List[str]],
        num_episodes: int = 1,
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_data: Dict with 'prompts' and 'completions' lists
            num_episodes: Number of evaluation passes (usually 1)

        Returns:
            Dictionary of evaluation metrics
        """
        # Create dataset and dataloader
        test_dataset = SFTDataset(
            prompts=test_data["prompts"],
            completions=test_data["completions"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                total_loss += outputs.loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches

        metrics = {
            "loss": avg_loss,
            "perplexity": np.exp(avg_loss),
        }

        print(f"Evaluation Results:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")

        return metrics

    def _validate(self, val_data: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Run validation on validation data.

        Args:
            val_data: Dict with 'prompts' and 'completions' lists

        Returns:
            Dictionary of validation metrics
        """
        # Sample if needed
        if len(val_data["prompts"]) > self.validation_num_samples:
            indices = np.random.choice(
                len(val_data["prompts"]), self.validation_num_samples, replace=False
            )
            sampled_data = {
                "prompts": [val_data["prompts"][i] for i in indices],
                "completions": [val_data["completions"][i] for i in indices],
            }
        else:
            sampled_data = val_data

        # Create dataset and dataloader
        val_dataset = SFTDataset(
            prompts=sampled_data["prompts"],
            completions=sampled_data["completions"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                total_loss += outputs.loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches

        return {"loss": avg_loss}

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> List[str]:
        """
        Generate completions for given prompts.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling

        Returns:
            List of generated completion strings
        """
        self.model.eval()
        completions = []

        with torch.no_grad():
            for prompt in prompts:
                # Tokenize
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate
                generated_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                # Extract completion
                prompt_len = inputs["input_ids"].shape[1]
                completion_ids = generated_ids[:, prompt_len:]
                completion = self.tokenizer.decode(
                    completion_ids[0], skip_special_tokens=True
                ).strip()

                completions.append(completion)

        return completions

    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
            "global_step": self.global_step,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)

        print(f"✓ Loaded checkpoint from {path}")
        print(f"  Resumed at step: {self.global_step}")
