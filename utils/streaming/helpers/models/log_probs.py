from dataclasses import dataclass
from typing import List


@dataclass
class LogProbs:
    tokens: List[int]
    token_logprobs: List[float]
    top_logprobs: List[List[float]]
    text_offset: List[int]