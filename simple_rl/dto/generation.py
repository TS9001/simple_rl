from dataclasses import dataclass
from typing import List
import torch


@dataclass
class GeneratedCompletionsResult:
    """Result from generating completions for a batch of prompts."""

    generated_ids: torch.Tensor
    generated_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    completion_texts: List[str]
    prompt_end_positions: torch.Tensor
    total_sequences: int
