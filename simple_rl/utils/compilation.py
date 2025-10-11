"""Model compilation utilities for PyTorch models."""

import torch
from typing import Dict, Any, Optional


class ModelCompilationManager:
    """Manages model compilation with device-specific optimizations."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize compilation manager.

        Args:
            config: Configuration dictionary with compilation settings
        """
        self.config = config
        self._compiled_device_type: Optional[str] = None

    def ensure_compiled(self, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
        """
        Compile model based on device and configuration.

        Args:
            model: PyTorch model to compile
            device: Target device

        Returns:
            Compiled model (or original if compilation fails/disabled)
        """
        # Check if compile is enabled in config
        compile_cfg = self.config.get("model", {}).get("compile", {})
        compile_enabled = compile_cfg.get("enabled", True)

        if not compile_enabled:
            self._compiled_device_type = "disabled"
            return model

        # Determine current target
        target_type = getattr(device, "type", None)

        # Skip compile on MPS by default (usually slower/unstable)
        if target_type == "mps":
            disable_mps_compile = compile_cfg.get("disable_on_mps", True)
            if disable_mps_compile:
                self._compiled_device_type = "disabled"
                return model

        if target_type == "cpu":
            # Defer compilation until the module is moved to an accelerated backend
            self._compiled_device_type = None
            return model

        if self._compiled_device_type in (target_type, "disabled", "failed"):
            return model

        compile_kwargs = self._get_compile_kwargs(target_type, compile_cfg)

        try:
            compiled_model = torch.compile(model, **compile_kwargs)
            compiled_model.to(device)
            self._compiled_device_type = target_type
            return compiled_model
        except Exception:
            self._compiled_device_type = "failed"
            return model

    def _get_compile_kwargs(self, device_type: str, compile_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Get device-specific compilation arguments."""
        compile_kwargs: Dict[str, Any] = {}

        if device_type == "mps":
            # Use the recommended AOT eager backend for MPS
            compile_kwargs["backend"] = compile_cfg.get("backend", "aot_eager")
        elif device_type == "cuda":
            # Use max-autotune for best perf on CUDA; enable cudagraphs if possible
            compile_kwargs["mode"] = compile_cfg.get("mode", "max-autotune")
            compile_kwargs["options"] = {
                "triton.cudagraphs": True,
                "shape_padding": True,
            }
        else:
            # Skip compile for CPU or unsupported devices
            self._compiled_device_type = "disabled"
            return {}

        return compile_kwargs

    def reset_compilation_state(self) -> None:
        """Reset compilation state (useful when moving between devices)."""
        self._compiled_device_type = None


def compile_model(model: torch.nn.Module, config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """
    Compile a model with device-specific optimizations.

    Args:
        model: PyTorch model to compile
        config: Configuration dictionary
        device: Target device

    Returns:
        Compiled model
    """
    manager = ModelCompilationManager(config)
    return manager.ensure_compiled(model, device)
