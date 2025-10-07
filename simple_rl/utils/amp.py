"""Mixed precision and automatic mixed precision (AMP) utilities."""

import torch
from contextlib import nullcontext
from functools import partial
from typing import Optional, Dict, Any, Callable, Union
from torch.cuda.amp import GradScaler


def get_dtype_from_config(dtype_config: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    """
    Convert dtype configuration to torch.dtype.

    Args:
        dtype_config: String ("fp16", "bf16", "fp32") or torch.dtype object

    Returns:
        Corresponding torch.dtype or None
    """
    if dtype_config is None:
        return None

    alias_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    if isinstance(dtype_config, str):
        return alias_map.get(dtype_config.lower())
    elif isinstance(dtype_config, torch.dtype):
        return dtype_config
    else:
        raise ValueError(f"Unsupported dtype specification: {dtype_config!r}")


class AMPConfig:
    """Configuration for automatic mixed precision training."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AMP configuration.

        Args:
            config: Configuration dictionary with mixed precision settings
        """
        self.config = config
        self._autocast: Optional[Callable] = None
        self._grad_scaler: Optional[GradScaler] = None
        self._amp_enabled = False
        self._amp_device: Optional[str] = None

        self._setup_amp()

    def _setup_amp(self) -> None:
        """Set up AMP configuration based on device and settings."""
        # Support both top-level and nested optimization.mixed_precision
        mp_config = self.config.get("mixed_precision")
        if mp_config is None:
            mp_config = self.config.get("optimization", {}).get("mixed_precision", {})
        if mp_config is None:
            mp_config = {}

        device_type = self._get_device_type()

        # Normalize mp_config
        if isinstance(mp_config, str):
            mp_config = {"mode": mp_config}
        elif not isinstance(mp_config, dict):
            mp_config = {"mode": "auto"}

        mode = mp_config.get("mode", "auto")
        mp_enabled = mp_config.get("enabled", True)

        if device_type == "cuda" and mp_enabled and mode != "off":
            self._setup_cuda_amp(mp_config)
        elif device_type == "mps" and mp_enabled:
            self._setup_mps_amp(mp_config, mode)
        else:
            self._setup_cpu_amp()

    def _get_device_type(self) -> str:
        """Get the device type from config or auto-detect."""
        device = self.config.get("device")
        if device is not None:
            return getattr(torch.device(device), "type", str(device))

        # Auto-detect
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _setup_cuda_amp(self, mp_config: Dict[str, Any]) -> None:
        """Set up CUDA mixed precision."""
        dtype_key = mp_config.get("dtype", "fp16")
        target_dtype = get_dtype_from_config(dtype_key)

        if target_dtype is None:
            target_dtype = torch.float16  # fallback

        self._autocast = partial(torch.cuda.amp.autocast, dtype=target_dtype)
        self._grad_scaler = GradScaler(
            enabled=True,
            growth_factor=mp_config.get("growth_factor", 2.0),
            backoff_factor=mp_config.get("backoff_factor", 0.5),
            growth_interval=mp_config.get("growth_interval", 2000),
        )
        self._amp_enabled = True
        self._amp_device = "cuda"

    def _setup_mps_amp(self, mp_config: Dict[str, Any], mode: str) -> None:
        """Set up MPS mixed precision (experimental)."""
        requested = mode in {"fp16", "mps_fp16", "enable"}

        if requested:
            try:
                dtype_key = mp_config.get("dtype", "fp16")
                target_dtype = get_dtype_from_config(dtype_key)

                if target_dtype is None:
                    target_dtype = torch.float16  # fallback

                self._autocast = partial(
                    torch.autocast, device_type="mps", dtype=target_dtype
                )
                self._amp_enabled = True
                self._amp_device = "mps"
            except RuntimeError:
                print(
                    "Mixed precision autocast on MPS failed to initialize. Falling back to full precision."
                )
                self._autocast = nullcontext()
                self._amp_enabled = False
        else:
            if mode not in {"off", "disable"}:
                print(
                    "Mixed precision on MPS remains experimental in the latest PyTorch nightly; keeping full precision."
                )
            self._autocast = nullcontext()
            self._amp_enabled = False

    def _setup_cpu_amp(self) -> None:
        """Set up CPU (no mixed precision)."""
        self._autocast = nullcontext()
        self._amp_enabled = False

    @property
    def autocast(self) -> Callable:
        """Get the autocast context manager."""
        return self._autocast if self._autocast is not None else nullcontext

    @property
    def grad_scaler(self) -> Optional[GradScaler]:
        """Get the gradient scaler for AMP."""
        return self._grad_scaler

    @property
    def enabled(self) -> bool:
        """Check if AMP is enabled."""
        return self._amp_enabled

    @property
    def device(self) -> Optional[str]:
        """Get the device type for AMP."""
        return self._amp_device


def create_amp_config(config: Dict[str, Any]) -> AMPConfig:
    """
    Create AMP configuration from config dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        AMPConfig instance
    """
    return AMPConfig(config)
