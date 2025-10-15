"""HuggingFace model wrappers and utilities."""

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from simple_rl.utils.device import get_target_device, apply_device_optimizations, clear_device_cache
from simple_rl.utils.model_loading import load_huggingface_model_and_tokenizer, setup_tokenizer_and_model_config


# Enable parallel tokenization on MPS (safe and faster), disable elsewhere
_use_parallel_tokenizers = (
    hasattr(torch.backends, "mps") and 
    torch.backends.mps.is_available()
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true" if _use_parallel_tokenizers else "false")


class LanguageModel(nn.Module):
    """
    Language model wrapper for HuggingFace causal language models.

    Handles text generation, log probability computation, and tokenization.
    Can be used by any algorithm that needs language generation capabilities.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize language model with HuggingFace model.

        Args:
            config: Configuration dictionary with model settings
            model: Optional pre-loaded HuggingFace model
            tokenizer: Optional pre-loaded HuggingFace tokenizer
        """
        super().__init__()

        self.config = config

        model_config = config.get("model", {})
        self.model_name = model_config.get("model_name")
        self.max_length = model_config.get("max_length", 512)

        # Determine target device
        target_device = get_target_device(model_config.get("device") or config.get("device"))

        # Load HuggingFace model and tokenizer using utility (if not provided)
        if model is None or tokenizer is None:
            self.model, self.tokenizer = load_huggingface_model_and_tokenizer(
                self.model_name, config
            )
        else:
            self.model = model
            self.tokenizer = tokenizer

        # Set up tokenizer and model config
        setup_tokenizer_and_model_config(self.model, self.tokenizer)

        # For decoder-only models, use left padding by default
        # But we'll switch to right padding for batched generation to avoid inf/nan issues
        self.tokenizer.padding_side = "left"

        # Get model config
        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size

        # Track optimizations
        self._using_bettertransformer: bool = False

        # Resolve initial device placement and apply optimizations
        super().to(target_device)
        apply_device_optimizations()

    def to(self, *args, **kwargs):
        """Override to() to re-run backend-specific setup after device moves."""

        module = super().to(*args, **kwargs)
        apply_device_optimizations()
        return module


    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.model.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the language model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        return outputs.logits

    def generate(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 400,  # Increased to 400 for longer CoT completions (avg 288 tokens)
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate completions for given prompts.

        Args:
            prompt_ids: Prompt token IDs [batch_size, prompt_len]
            attention_mask: Attention mask for prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Tuple of (generated_ids, attention_mask)
        """
        # Use default eos_token_id unless overridden in kwargs
        if 'eos_token_id' not in kwargs:
            kwargs['eos_token_id'] = self.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=False,
                **kwargs,
            )

        generated_ids = outputs.sequences

        # Create attention mask for generated sequence (ensure it's on the same device)
        generated_attention_mask = (
            (generated_ids != self.tokenizer.pad_token_id)
            .long()
            .to(generated_ids.device)
        )

        return generated_ids, generated_attention_mask

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        logits_to_keep: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute log probabilities for a sequence.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            logits_to_keep: If specified, only compute log probs for the last N tokens
                          (model still sees full context, this only affects output size)

        Returns:
            Log probabilities [batch_size, seq_len-1] or [batch_size, logits_to_keep]
            Note: Padding positions (where attention_mask=0) will have log_prob=0.0
        """
        original_seq_len = input_ids.size(1)
        left_pad_removed = 0

        # OPTIMIZATION: Remove left padding before computing log probs to save computation
        if attention_mask is not None:
            # Find where content starts for each sequence (first non-zero in mask)
            first_valid = (attention_mask != 0).int().argmax(dim=1)  # [batch_size]
            min_left_pad = first_valid.min().item()

            if min_left_pad > 0:
                # Remove common left padding from all sequences
                input_ids = input_ids[:, min_left_pad:]
                attention_mask = attention_mask[:, min_left_pad:]
                left_pad_removed = min_left_pad

        # Get logits from model (full forward pass with complete context)
        logits = self.forward(input_ids, attention_mask=attention_mask)

        # Shift logits and labels for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Also shift attention mask to align with shifted labels
        if attention_mask is not None:
            shift_attention_mask = attention_mask[:, 1:].contiguous()

        # If logits_to_keep specified, only keep last N positions
        # (saves memory in log_softmax and gather operations)
        if logits_to_keep is not None:
            actual_seq_len = shift_logits.size(1)
            logits_to_keep = min(logits_to_keep, actual_seq_len)

            if logits_to_keep < actual_seq_len:
                # Use explicit indexing instead of negative indexing for MPS compatibility
                start_idx = actual_seq_len - logits_to_keep
                shift_logits = shift_logits[:, start_idx:, :]
                shift_labels = shift_labels[:, start_idx:]
                if attention_mask is not None:
                    shift_attention_mask = shift_attention_mask[:, start_idx:]

        # MPS WORKAROUND: Compute log probs on CPU for large vocab sizes
        # MPS has numerical precision issues with large vocabulary log_softmax + gather operations
        # This causes incorrect zero values in computed log probabilities
        if shift_logits.device.type == 'mps' and shift_logits.size(-1) > 50000:
            original_device = shift_logits.device

            # Move to CPU for computation
            shift_logits_cpu = shift_logits.to('cpu')
            shift_labels_cpu = shift_labels.to('cpu')

            # Compute on CPU
            log_probs_all_cpu = F.log_softmax(shift_logits_cpu, dim=-1)
            log_probs_cpu = torch.gather(
                log_probs_all_cpu, dim=-1, index=shift_labels_cpu.unsqueeze(-1)
            ).squeeze(-1)

            # Move result back to original device
            log_probs = log_probs_cpu.to(original_device)

            # Clean up CPU tensors
            del shift_logits_cpu, shift_labels_cpu, log_probs_all_cpu, log_probs_cpu
        else:
            # Standard computation on device
            log_probs_all = F.log_softmax(shift_logits, dim=-1)
            log_probs = torch.gather(
                log_probs_all, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

        # CRITICAL FIX: Mask out padding positions
        # Set log probs to 0.0 for padding tokens (where attention_mask=0)
        if attention_mask is not None:
            log_probs = log_probs * shift_attention_mask.float()

        return log_probs

    def tokenize(
        self,
        texts: List[str],
        padding_side: str = "right",
        return_tensors: str = "pt",

    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text strings.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            truncation: Whether to truncate
            return_tensors: Return type ("pt" for PyTorch tensors)

        Returns:
            Dictionary with input_ids and attention_mask
        """


        tokenized = self.tokenizer(
            texts, return_tensors=return_tensors, padding=True, padding_side=padding_side
        )

        return tokenized

    def decode(
        self, token_ids: torch.Tensor, skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs [batch_size, seq_len]
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded text strings
        """
        return self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def get_prompt_length(self, prompt_ids: torch.Tensor) -> int:
        """
        Get the length of prompt in tokens.

        Args:
            prompt_ids: Prompt token IDs [batch_size, prompt_len]

        Returns:
            Length of prompt
        """
        return prompt_ids.shape[1]
