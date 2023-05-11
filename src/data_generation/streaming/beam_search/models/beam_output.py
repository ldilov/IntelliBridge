from dataclasses import dataclass
from typing import List, Tuple

import torch

from src.data_generation.streaming.beam_search.models.sampled_token import SampledToken
from src.data_generation.streaming.beam_search.models.token_probs import TokenProbsResult


@dataclass
class BeamOutput:
    token_probs: TokenProbsResult
    reason: str
    past_key_values: Tuple

    def __iter__(self):
        yield self.token_probs
        yield self.reason
        yield self.past_key_values

    def tolist(self) -> List:
        return [self.token_probs, self.reason, self.past_key_values]

    @property
    def token_id(self):
        beam_output_token: SampledToken
        _, beam_output_token, top_k_probs = self.token_probs

        output_token_id: torch.Tensor
        adjusted_prob: torch.Tensor
        output_token_id, _ = beam_output_token

        return output_token_id
