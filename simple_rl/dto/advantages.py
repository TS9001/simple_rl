from dataclasses import dataclass


@dataclass
class AdvantageStats:
    """Statistics about computed advantages including raw and clamped values."""

    advantages_min_raw: float
    advantages_max_raw: float
    advantages_min: float
    advantages_max: float
    advantages_clamped_low_frac: float
    advantages_clamped_high_frac: float
