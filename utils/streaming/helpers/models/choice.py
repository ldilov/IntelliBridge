from dataclasses import dataclass
from typing import Optional

from utils.streaming.helpers.models.log_probs import LogProbs


@dataclass
class Choice:
    text: str
    index: int
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[str] = None