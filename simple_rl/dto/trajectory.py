from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class TrajectoryBatch:
    """Batch of generated trajectories with rewards and metadata."""

    prompts: Optional[List[str]]
    completions: Optional[List[str]]
    rewards: torch.Tensor
    completion_mask: List[torch.Tensor]
    format_rewards: torch.Tensor
    correctness_rewards: torch.Tensor
    generated_ids: List[torch.Tensor]
    attention_mask: List[torch.Tensor]
    prompt_end_positions: torch.Tensor
