from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class CheckpointData:
    """Complete checkpoint containing model, optimizer, and training state."""

    policy_state_dict: Dict
    ref_policy_state_dict: Optional[Dict]
    optimizer_state_dict: Dict
    scheduler_state_dict: Optional[Dict]
    scaler_state_dict: Optional[Dict]
    config: Dict[str, Any]
    total_steps: int
    episode: int
    current_episode: int
