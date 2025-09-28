"""Device management utilities for PyTorch models."""

import torch
from typing import Optional, Dict, Any


def get_target_device(device_config: Optional[str] = None) -> torch.device:
    """
    Determine the target device based on configuration and availability.

    Args:
        device_config: Device specification from config (e.g., "cuda", "mps", "cpu")

    Returns:
        PyTorch device object
    """
    if device_config is not None:
        return torch.device(device_config)

    # Auto-select device based on availability
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_type(device: torch.device) -> str:
    """Get the type of a device (cuda, mps, cpu)."""
    return getattr(device, "type", str(device))


def apply_device_optimizations() -> None:
    """Apply backend-specific performance optimizations."""
    # Global optimizations
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # CUDA-specific optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except AttributeError:
            pass

        # Clear cache
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

        # Set memory fraction
        if hasattr(torch.cuda, "set_per_process_memory_fraction"):
            try:
                torch.cuda.set_per_process_memory_fraction(0.95)
            except Exception:
                pass

    # MPS-specific optimizations
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.backends.mps.matmul.allow_tf32 = True
        except AttributeError:
            pass

        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass


def clear_device_cache(device: Optional[torch.device] = None) -> None:
    """Clear cache for the specified device or auto-detect."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_type = get_device_type(device)

    if device_type == "cuda" and hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()
    elif device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass


class DeviceManager:
    """Context manager for device operations with automatic optimization."""

    def __init__(self, device_config: Optional[str] = None):
        self.device = get_target_device(device_config)
        self._applied_optimizations = False

    def __enter__(self):
        if not self._applied_optimizations:
            apply_device_optimizations()
            self._applied_optimizations = True
        return self.device

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clear cache on exit
        clear_device_cache(self.device)
