from dataclasses import dataclass
from typing import List


@dataclass
class BatchData:
    """Batch input data container."""

    prompts: List[str]
    answers: List[str]
