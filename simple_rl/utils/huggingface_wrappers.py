import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from simple_rl.utils.model_loading import load_huggingface_model_and_tokenizer, setup_tokenizer_and_model_config


_use_parallel_tokenizers = (
    hasattr(torch.backends, "mps") and
    torch.backends.mps.is_available()
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true" if _use_parallel_tokenizers else "false")


class LanguageModel(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        super().__init__()

        self.config = config

        model_config = config.get("model", {})
        self.model_name = model_config.get("model_name")
        self.max_length = model_config.get("max_length", 512)

        device_config = model_config.get("device") or config.get("device")
        if device_config:
            target_device = torch.device(device_config)
        elif torch.cuda.is_available():
            target_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            target_device = torch.device("mps")
        else:
            target_device = torch.device("cpu")

        # Load HuggingFace model and tokenizer using utility (if not provided)
        if model is None or tokenizer is None:
            self.model, self.tokenizer = load_huggingface_model_and_tokenizer(
                self.model_name, config
            )
        else:
            self.model = model
            self.tokenizer = tokenizer

        setup_tokenizer_and_model_config(self.model, self.tokenizer)

        self.tokenizer.padding_side = "left"

        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size

        self._using_bettertransformer: bool = False
        self._compiled: bool = False

        super().to(target_device)

        compile_config = model_config.get("compile", {})
        if compile_config.get("enabled", False):
            try:
                backend = compile_config.get("backend", "inductor")
                mode = compile_config.get("mode", "default")

                print(f"🔥 Compiling model with torch.compile (backend={backend}, mode={mode})...")
                print(f"   Note: First forward pass will be slow (compilation), then ~20-30% faster")

                self.model = torch.compile(
                    self.model,
                    backend=backend,
                    mode=mode,
                    fullgraph=False,
                )
                self._compiled = True
                print(f"   ✓ Model compiled successfully")

            except Exception as e:
                print(f"   ⚠️  Compilation failed: {e}")
                print(f"   Continuing without compilation...")
                self._compiled = False

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        return outputs.logits

    def generate(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 400,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        logits = self.forward(input_ids, attention_mask=attention_mask)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        del logits

        if attention_mask is not None:
            shift_attention_mask = attention_mask[:, 1:].contiguous()

        if logits_to_keep is not None:
            actual_seq_len = shift_logits.size(1)
            logits_to_keep = min(logits_to_keep, actual_seq_len)

            if logits_to_keep < actual_seq_len:
                start_idx = actual_seq_len - logits_to_keep
                shift_logits = shift_logits[:, start_idx:, :]
                shift_labels = shift_labels[:, start_idx:]
                if attention_mask is not None:
                    shift_attention_mask = shift_attention_mask[:, start_idx:]

        log_probs_all = F.log_softmax(shift_logits, dim=-1)

        del shift_logits

        log_probs = torch.gather(
            log_probs_all, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1).float()

        del log_probs_all

        if attention_mask is not None:
            log_probs = log_probs * shift_attention_mask.float()

        return log_probs

    def tokenize(
        self,
        texts: List[str],
        padding_side: str = "right",
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            texts, return_tensors=return_tensors, padding=True, padding_side=padding_side
        )
        return tokenized

    def decode(
        self, token_ids: torch.Tensor, skip_special_tokens: bool = True
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def get_prompt_length(self, prompt_ids: torch.Tensor) -> int:
        return prompt_ids.shape[1]
