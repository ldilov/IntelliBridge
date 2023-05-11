from dataclasses import dataclass
from typing import Tuple

import torch

from src.data_generation.streaming.beam_search.models.beam_output import BeamOutput
from src.data_generation.streaming.beam_search.models.sampled_token import SampledToken
from src.data_generation.streaming.beam_search.models.token_probs import TokenProbsResult
from src.data_generation.streaming.stream_tokenizer import StreamTokenizer


@dataclass
class BeamOutputData:
    output_token_id: torch.Tensor
    adjusted_prob: torch.Tensor
    reason: str
    decoded_output: str
    top_k_probs: torch.Tensor
    past_key_values: Tuple

    def __init__(self, beam_output: BeamOutput, decoded_output: str):
        super().__init__()

        self._past_key_values = None
        self._top_k_probs = None

        token_probs_result: TokenProbsResult
        reason: str
        past_key_values: Tuple
        token_probs_result, self.reason, self.past_key_values = beam_output

        beam_output_token: SampledToken
        _, beam_output_token, self.top_k_probs = token_probs_result

        output_token_id: torch.Tensor
        adjusted_prob: torch.Tensor
        self.output_token_id, self.adjusted_prob = beam_output_token

        self.decoded_output = decoded_output

    def __iter__(self):
        yield self.output_token_id
        yield self.adjusted_prob
        yield self.reason
        yield self.decoded_output

    def __lt__(self, comparable: 'BeamOutputData'):
        if isinstance(comparable, BeamOutputData):
            return self.adjusted_prob < comparable.adjusted_prob

        type_str = type(comparable).__name__
        return NotImplemented(f"Cannot compare BeamOutputData with other types. Current comparable type: {type_str}")

    def __str__(self):
        return self.decoded_output

    @property
    def token_id(self) -> torch.Tensor:
        return self.output_token_id

    @property
    def past_key_values(self) -> Tuple:
        return self._past_key_values

    @past_key_values.setter
    def past_key_values(self, value: Tuple):
        self._past_key_values = value

    @property
    def top_k_probs(self) -> torch.Tensor:
        return self._top_k_probs

    @top_k_probs.setter
    def top_k_probs(self, value: torch.Tensor):
        self._top_k_probs = value


