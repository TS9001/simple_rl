"""Checkpoint management utilities."""

import torch
from pathlib import Path
from typing import Dict, Any, Optional


def save_checkpoint(
    path: str,
    policy_state_dict: Dict[str, Any],
    ref_policy_state_dict: Optional[Dict[str, Any]] = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    total_steps: int = 0,
    episode: int = 0,
    **additional_metadata
) -> None:
    """
    Save model checkpoint.

    Args:
        path: Path to save checkpoint
        policy_state_dict: Policy model state dict
        ref_policy_state_dict: Reference policy state dict (optional)
        optimizer_state_dict: Optimizer state dict (optional)
        config: Configuration dictionary (optional)
        total_steps: Total training steps
        episode: Current episode
        **additional_metadata: Additional metadata to save
    """
    checkpoint = {
        "policy_state_dict": policy_state_dict,
        "total_steps": total_steps,
        "episode": episode,
    }

    if ref_policy_state_dict is not None:
        checkpoint["ref_policy_state_dict"] = ref_policy_state_dict

    if optimizer_state_dict is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state_dict

    if config is not None:
        checkpoint["config"] = config

    if additional_metadata:
        checkpoint["metadata"] = additional_metadata

    # Create directory if it doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint file
        device: Device to map tensors to

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


class CheckpointManager:
    """Manager for saving and loading checkpoints."""

    def __init__(self, save_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        policy_state_dict: Dict[str, Any],
        ref_policy_state_dict: Optional[Dict[str, Any]] = None,
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        total_steps: int = 0,
        episode: int = 0,
        filename: Optional[str] = None,
        **additional_metadata
    ) -> str:
        """
        Save checkpoint with automatic filename generation.

        Args:
            policy_state_dict: Policy model state dict
            ref_policy_state_dict: Reference policy state dict (optional)
            optimizer_state_dict: Optimizer state dict (optional)
            config: Configuration dictionary (optional)
            total_steps: Total training steps
            episode: Current episode
            filename: Custom filename (auto-generated if None)
            **additional_metadata: Additional metadata to save

        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_episode_{episode}_steps_{total_steps}.pt"

        filepath = self.save_dir / filename
        save_checkpoint(
            str(filepath),
            policy_state_dict,
            ref_policy_state_dict,
            optimizer_state_dict,
            config,
            total_steps,
            episode,
            **additional_metadata
        )

        return str(filepath)

    def load(self, filename: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Load checkpoint by filename.

        Args:
            filename: Checkpoint filename
            device: Device to map tensors to

        Returns:
            Checkpoint dictionary
        """
        filepath = self.save_dir / filename
        return load_checkpoint(str(filepath), device)

    def list_checkpoints(self) -> list:
        """List all checkpoint files in save directory."""
        return [f.name for f in self.save_dir.glob("*.pt")]

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the latest checkpoint filename."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        # Sort by modification time (newest first)
        checkpoint_paths = [self.save_dir / cp for cp in checkpoints]
        latest_checkpoint = max(checkpoint_paths, key=lambda p: p.stat().st_mtime)
        return latest_checkpoint.name
