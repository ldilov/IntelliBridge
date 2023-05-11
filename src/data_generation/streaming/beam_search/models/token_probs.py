from dataclasses import dataclass

import torch

from src.data_generation.streaming.beam_search.models.sampled_token import SampledToken


@dataclass
class TokenProbsResult:
    probs: torch.Tensor
    sampled_token: SampledToken
    top_k_probs: torch.Tensor

    def __iter__(self):
        yield self.probs
        yield self.sampled_token
        yield self.top_k_probs
