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
import math

from .base import BaseAlgorithm
from simple_rl.utils.logging_utils import create_logger


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
        debug: bool = False,
        mask_prompt: bool = True,
    ):
        """
        Initialize SFT dataset.

        Args:
            prompts: List of prompt strings
            completions: List of completion strings (targets)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            debug: Enable debug output
            mask_prompt: If True, mask prompt tokens in loss (standard).
                        If False, train on full sequence including prompt (like reference notebook).
        """
        self.prompts = prompts
        self.completions = completions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.debug = debug
        self.mask_prompt = mask_prompt
        self._debug_logged = False

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
            add_special_tokens=True,
        )
        prompt_length = len(prompt_encoding["input_ids"])

        labels = encoding["input_ids"].clone()

        # Calculate padding offset for left-padded sequences
        seq_len = encoding["input_ids"].size(1)
        nonpad_len = int(encoding["attention_mask"][0].sum().item())
        pad_len = seq_len - nonpad_len

        if self.mask_prompt:
            # Mask prompt tokens at the correct offset when using left padding
            start = pad_len  # where real tokens start
            end = min(start + prompt_length, seq_len)
            labels[0, start:end] = -100  # mask only the prompt span

        # Always mask padding tokens
        labels[0, encoding["attention_mask"][0] == 0] = -100

        if self.debug and not self._debug_logged:
            print("\n" + "=" * 80)
            print("[SFTDataset Debug - Full Training Example]")
            print("=" * 80)

            # Show the full message
            print("\n📝 FULL TRAINING TEXT:")
            print("-" * 80)
            print(full_text)
            print("-" * 80)

            print(f"\n📊 BREAKDOWN:")
            print(
                f"  Prompt: {repr(prompt[:100])}..."
                if len(prompt) > 100
                else f"  Prompt: {repr(prompt)}"
            )
            print(
                f"  Completion: {repr(completion[:100])}..."
                if len(completion) > 100
                else f"  Completion: {repr(completion)}"
            )
            print(f"  Mask prompt tokens: {self.mask_prompt}")

            # Tokenization details
            full_ids = encoding["input_ids"].squeeze(0)
            labels_flat = labels.squeeze(0)
            completion_indices = (labels_flat != -100).nonzero(as_tuple=True)

            print(f"\n🔢 TOKENIZATION:")
            print(f"  Total tokens: {len(full_ids)}")
            print(f"  Prompt tokens: {prompt_length}")
            print(
                f"  Non-masked tokens (for loss): {(labels_flat != -100).sum().item()}"
            )

            if completion_indices[0].numel() > 0:
                first_completion_idx = completion_indices[0][0].item()
                context_start = max(0, first_completion_idx - 3)
                context_end = min(len(full_ids), first_completion_idx + 5)
                context_ids = full_ids[context_start:context_end]
                first_completion_token = self.tokenizer.decode(
                    [full_ids[first_completion_idx]], skip_special_tokens=False
                )
                context_text = self.tokenizer.decode(
                    context_ids, skip_special_tokens=False
                )

                print(f"\n🎯 LABEL MASKING:")
                print(f"  First non-masked token index: {first_completion_idx}")
                print(f"  First non-masked token: {repr(first_completion_token)}")
                print(f"  Context around split: {repr(context_text)}")

            print("=" * 80 + "\n")

            self._debug_logged = True

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
            # Note: Logger not yet initialized, print is acceptable here
            print(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Ensure consistent padding behavior for decoder-only models
            self.tokenizer.padding_side = "left"
        else:
            self.tokenizer = tokenizer

        # Load model
        if model is None:
            model_name = config["model"]["model_name"]
            model_type = config.get("model", {}).get("model_type", "auto")

            # Determine dtype based on config (not just hardware availability)
            if model_type == "fp16":
                dtype = torch.float16
            elif model_type == "bf16":
                dtype = torch.bfloat16
            elif model_type == "fp32":
                dtype = torch.float32
            else:  # "auto"
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # Note: Logger not yet initialized, print is acceptable here
            print(f"Loading model: {model_name}")
            print(f"Model dtype: {dtype} (from config model_type='{model_type}')")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
        else:
            self.model = model

        self.model = self.model.to(self.device)

        # Ensure model config is aligned with tokenizer for padding/eos
        try:
            if getattr(self.model, "config", None) is not None:
                if getattr(self.model.config, "pad_token_id", None) is None:
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
                # Optionally disable cache if using gradient checkpointing
        except Exception:
            # Be conservative if model lacks expected attributes
            pass

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
        self.enable_gradient_checkpointing = training_config.get(
            "gradient_checkpointing", False
        )
        self.weight_decay = training_config.get("weight_decay", 0.01)
        self.label_smoothing = training_config.get("label_smoothing", 0.0)
        # Mixed precision (MPS autocast)
        self.mixed_precision = training_config.get("mixed_precision", {}) or {}
        self.use_mps_autocast = (
            isinstance(self.device, torch.device)
            and self.device.type == "mps"
            and bool(self.mixed_precision.get("enabled", False))
        )
        _dtype_key = str(self.mixed_precision.get("dtype", "fp16")).lower()
        self.mps_autocast_dtype = (
            torch.float16 if _dtype_key in ("fp16", "float16") else torch.bfloat16
        )

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
        self.validation_seed = validation_config.get("seed", 42)
        self._val_sample_indices = None

        # Initialize optimizer (will be set up in train())
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.global_step = 0

        # Initialize logger
        self.logger = create_logger(self.config)
        if use_wandb:
            self.logger.init_wandb()

        # Initialize wandb if requested (legacy support)
        if use_wandb:
            import wandb

            wandb_config = config.get("wandb", {})
            wandb.init(
                project=wandb_config.get("project", "simple-rl-sft"),
                config=config,
                name=wandb_config.get("run_name", None),
            )

        self.logger.info(
            f"✓ SFT initialized with model: {config['model']['model_name']}"
        )
        self.logger.info(
            f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        self.logger.info(f"  Device: {self.device}")

        # Check for NaN/Inf in model weights (critical safety check)
        self._check_model_weights_sanity()

    def _check_model_weights_sanity(self):
        """Check if model weights contain NaN or Inf values."""
        nan_params = []
        inf_params = []

        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)

        if nan_params or inf_params:
            self.logger.error("\n" + "=" * 80)
            self.logger.error("🚨 CORRUPT MODEL WEIGHTS DETECTED")
            self.logger.error("=" * 80)
            if nan_params:
                self.logger.error(
                    f"\nParameters with NaN values ({len(nan_params)} total):"
                )
                for name in nan_params[:10]:  # Show first 10
                    self.logger.error(f"  - {name}")
                if len(nan_params) > 10:
                    self.logger.error(f"  ... and {len(nan_params) - 10} more")
            if inf_params:
                self.logger.error(
                    f"\nParameters with Inf values ({len(inf_params)} total):"
                )
                for name in inf_params[:10]:  # Show first 10
                    self.logger.error(f"  - {name}")
                if len(inf_params) > 10:
                    self.logger.error(f"  ... and {len(inf_params) - 10} more")
            self.logger.error("\n💡 Possible causes:")
            self.logger.error("  1. Corrupted checkpoint file")
            self.logger.error(
                "  2. Previous training run had NaN loss that corrupted weights"
            )
            self.logger.error("  3. Incorrect dtype conversion (e.g., fp16 overflow)")
            self.logger.error("  4. Model loading error")
            self.logger.error("\n🔧 Solutions:")
            self.logger.error(
                "  1. Delete corrupted checkpoints and retrain from pretrained model"
            )
            self.logger.error("  2. Use fp32 instead of fp16 for numerical stability")
            self.logger.error("  3. Check model loading code for dtype issues")
            self.logger.error("=" * 80)
            raise ValueError(
                f"Model contains NaN/Inf weights! "
                f"NaN params: {len(nan_params)}, Inf params: {len(inf_params)}. "
                f"Cannot proceed with training."
            )

        self.logger.info("  ✓ Model weights are clean (no NaN/Inf)")

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
            debug=train_data.get(
                "debug_boundary",
                self.config.get("training", {}).get("debug_boundary", False),
            ),
            mask_prompt=self.config.get("training", {}).get("mask_prompt", True),
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Calculate total training steps (ceil to avoid under-counting)
        num_epochs = num_episodes or self.num_epochs
        updates_per_epoch = (
            math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
            if len(train_dataloader) > 0
            else 0
        )
        num_training_steps = updates_per_epoch * num_epochs

        # Initialize optimizer if not already done
        if self.optimizer is None:
            # Weight decay with no_decay for biases and layer norms
            decay_params = []
            no_decay_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(
                    nd in name
                    for nd in [
                        "bias",
                        "LayerNorm.weight",
                        "layer_norm.weight",
                        "ln_f.weight",
                        "ln_attn.weight",
                    ]
                ):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            optimizer_grouped_parameters = [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]

            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        # Initialize scheduler if not already done
        if self.scheduler is None:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=num_training_steps,
            )

        # Enable gradient checkpointing for memory/stability if requested
        if self.enable_gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            if getattr(self.model, "config", None) is not None:
                # use_cache must be disabled when using gradient checkpointing
                if hasattr(self.model.config, "use_cache"):
                    self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

        self.logger.info(f"Starting SFT training for {num_epochs} epochs...")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(
            f"Gradient accumulation steps: {self.gradient_accumulation_steps}"
        )
        self.logger.info(
            f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}"
        )
        self.logger.info("=" * 50)

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

        # Initialize gradients
        self.optimizer.zero_grad()

        for epoch in range(num_epochs):
            epoch_loss_sum = 0.0  # sum of unscaled (pre-accumulation) losses
            epoch_start_time = time.time()

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                batch_start_time = time.time()

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                if self.use_mps_autocast:
                    with torch.autocast(
                        device_type="mps", dtype=self.mps_autocast_dtype
                    ):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        if self.label_smoothing and self.label_smoothing > 0.0:
                            logits = outputs.logits
                            shift_logits = logits[:, :-1, :].contiguous()
                            shift_labels = labels[:, 1:].contiguous()
                            loss_fct = nn.CrossEntropyLoss(
                                ignore_index=-100,
                                label_smoothing=float(self.label_smoothing),
                            )
                            loss = loss_fct(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                            )
                        else:
                            loss = outputs.loss
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    if self.label_smoothing and self.label_smoothing > 0.0:
                        logits = outputs.logits
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        loss_fct = nn.CrossEntropyLoss(
                            ignore_index=-100,
                            label_smoothing=float(self.label_smoothing),
                        )
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )
                    else:
                        loss = outputs.loss

                # Scale loss for gradient accumulation
                raw_loss_value = float(loss.item())

                # NaN detection - fail fast with helpful error message
                if math.isnan(raw_loss_value) or math.isinf(raw_loss_value):
                    self.logger.error("\n" + "=" * 80)
                    self.logger.error("🚨 NaN/Inf LOSS DETECTED")
                    self.logger.error("=" * 80)
                    self.logger.error(f"Epoch: {epoch+1}/{num_epochs}")
                    self.logger.error(f"Batch: {batch_idx+1}/{len(train_dataloader)}")
                    self.logger.error(f"Global step: {self.global_step}")
                    self.logger.error(f"Loss value: {raw_loss_value}")
                    self.logger.error(
                        f"Learning rate: {self.scheduler.get_last_lr()[0]:.2e}"
                    )
                    self.logger.error(
                        f"Model dtype: {next(self.model.parameters()).dtype}"
                    )
                    self.logger.error("\n💡 Common causes:")
                    self.logger.error("  1. Learning rate too high (try 1e-6 to 5e-6)")
                    self.logger.error(
                        "  2. Using fp16 without mixed precision training"
                    )
                    self.logger.error("  3. Gradient explosion (check gradient norms)")
                    self.logger.error("  4. Bad data (inf/nan in input)")
                    self.logger.error("\n📊 Debug info:")
                    self.logger.error(f"  Input shape: {input_ids.shape}")
                    self.logger.error(
                        f"  Input has nan: {torch.isnan(input_ids.float()).any().item()}"
                    )
                    self.logger.error(f"  Labels shape: {labels.shape}")
                    self.logger.error(
                        f"  Non-masked labels: {(labels != -100).sum().item()}"
                    )
                    self.logger.error("=" * 80)

                    raise ValueError(
                        f"NaN/Inf loss detected at step {self.global_step}. "
                        f"Loss={raw_loss_value}. See logs above for debugging info."
                    )

                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                # Track unscaled loss for accurate epoch averaging
                epoch_loss_sum += raw_loss_value

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
                    training_metrics["loss"].append(raw_loss_value)
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
                        self.logger.info(
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
                        self.logger.info(f"\n{'='*60}")
                        self.logger.info(f"VALIDATION AT STEP {self.global_step}")
                        self.logger.info(f"{'='*60}")

                        val_metrics = self._validate(val_data)

                        validation_metrics["step"].append(self.global_step)
                        validation_metrics["loss"].append(val_metrics["loss"])

                        self.logger.info(f"Validation Loss: {val_metrics['loss']:.4f}")

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
                        self.logger.info(f"  → Saved checkpoint to {checkpoint_path}")

            # Flush any remaining gradients if last batch didn't trigger update
            # This handles edge case where len(dataloader) % gradient_accumulation_steps != 0
            if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
                self.logger.info(
                    f"\nFlushing remaining gradients from last {(batch_idx + 1) % self.gradient_accumulation_steps} batches..."
                )
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = (
                epoch_loss_sum / len(train_dataloader)
                if len(train_dataloader) > 0
                else float("nan")
            )
            self.logger.info(
                f"\nEpoch {epoch+1} completed in {epoch_time:.2f}s | Avg Loss: {avg_epoch_loss:.4f}"
            )
            self.logger.info("=" * 50)
        # Training complete
        total_training_time = time.time() - training_start_time
        self.logger.info("=" * 50)
        self.logger.info("Training complete!")
        self.logger.info(
            f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)"
        )
        self.logger.info(f"Total steps: {self.global_step}")

        # Always save final checkpoint
        final_checkpoint_path = (
            checkpoint_dir / f"checkpoint_step_{self.global_step}_final.pt"
        )
        self.save_checkpoint(str(final_checkpoint_path))
        self.logger.info(f"  → Saved final checkpoint to {final_checkpoint_path}")

        return {
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "total_time": total_training_time,
            "final_loss": (
                training_metrics["loss"][-1] if training_metrics["loss"] else None
            ),
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
            # Initialize optimizer if needed (match HF Trainer defaults)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
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
        # IMPORTANT: Use same mask_prompt setting as training!
        test_dataset = SFTDataset(
            prompts=test_data["prompts"],
            completions=test_data["completions"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mask_prompt=self.config.get("training", {}).get("mask_prompt", True),
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

                if self.use_mps_autocast:
                    with torch.autocast(
                        device_type="mps", dtype=self.mps_autocast_dtype
                    ):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        if self.label_smoothing and self.label_smoothing > 0.0:
                            logits = outputs.logits
                            shift_logits = logits[:, :-1, :].contiguous()
                            shift_labels = labels[:, 1:].contiguous()
                            loss_fct = nn.CrossEntropyLoss(
                                ignore_index=-100,
                                label_smoothing=float(self.label_smoothing),
                            )
                            loss = loss_fct(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                            )
                            total_loss += loss.item()
                        else:
                            total_loss += outputs.loss.item()
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    if self.label_smoothing and self.label_smoothing > 0.0:
                        logits = outputs.logits
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        loss_fct = nn.CrossEntropyLoss(
                            ignore_index=-100,
                            label_smoothing=float(self.label_smoothing),
                        )
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )
                        total_loss += loss.item()
                    else:
                        total_loss += outputs.loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches

        metrics = {
            "loss": avg_loss,
            "perplexity": np.exp(avg_loss),
        }

        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"  Loss: {metrics['loss']:.4f}")
        self.logger.info(f"  Perplexity: {metrics['perplexity']:.2f}")

        return metrics

    def _validate(self, val_data: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Run validation on validation data.

        Args:
            val_data: Dict with 'prompts' and 'completions' lists

        Returns:
            Dictionary of validation metrics
        """
        # Sample deterministically if needed (persist indices across validations)
        if len(val_data["prompts"]) > self.validation_num_samples:
            if self._val_sample_indices is None:
                rng = np.random.RandomState(self.validation_seed)
                self._val_sample_indices = rng.choice(
                    len(val_data["prompts"]), self.validation_num_samples, replace=False
                )
            indices = self._val_sample_indices
            sampled_data = {
                "prompts": [val_data["prompts"][i] for i in indices],
                "completions": [val_data["completions"][i] for i in indices],
            }
        else:
            sampled_data = val_data

        # Create dataset and dataloader
        # IMPORTANT: Use same mask_prompt setting as training!
        val_dataset = SFTDataset(
            prompts=sampled_data["prompts"],
            completions=sampled_data["completions"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mask_prompt=self.config.get("training", {}).get("mask_prompt", True),
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

                if self.use_mps_autocast:
                    with torch.autocast(
                        device_type="mps", dtype=self.mps_autocast_dtype
                    ):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        if self.label_smoothing and self.label_smoothing > 0.0:
                            logits = outputs.logits
                            shift_logits = logits[:, :-1, :].contiguous()
                            shift_labels = labels[:, 1:].contiguous()
                            loss_fct = nn.CrossEntropyLoss(
                                ignore_index=-100,
                                label_smoothing=float(self.label_smoothing),
                            )
                            loss = loss_fct(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                            )
                            total_loss += loss.item()
                        else:
                            total_loss += outputs.loss.item()
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    if self.label_smoothing and self.label_smoothing > 0.0:
                        logits = outputs.logits
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        loss_fct = nn.CrossEntropyLoss(
                            ignore_index=-100,
                            label_smoothing=float(self.label_smoothing),
                        )
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )
                        total_loss += loss.item()
                    else:
                        total_loss += outputs.loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches

        return {"loss": avg_loss}

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 400,  # Increased to 400 for longer CoT completions (avg 288 tokens)
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        batch_size: int = 8,  # Batch size for generation
    ) -> List[str]:
        """
        Generate completions for given prompts using batched generation.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            batch_size: Number of prompts to process in parallel

        Returns:
            List of generated completion strings
        """
        self.model.eval()
        all_completions = []

        with torch.no_grad():
            # Process prompts in batches for speed
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]

                # Tokenize batch with padding
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate for batch
                generated_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                # Extract completions for each item in batch
                for j, gen_ids in enumerate(generated_ids):
                    # Find where the prompt ends (first non-pad token in input)
                    prompt_len = (inputs["attention_mask"][j] == 1).sum().item()

                    # Extract only the completion part
                    completion_ids = gen_ids[prompt_len:]
                    completion = self.tokenizer.decode(
                        completion_ids, skip_special_tokens=True
                    ).strip()

                    all_completions.append(completion)

        return all_completions

    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer else None
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
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

        self.logger.info(f"✓ Loaded checkpoint from {path}")
        self.logger.info(f"  Resumed at step: {self.global_step}")
