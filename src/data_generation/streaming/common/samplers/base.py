from abc import abstractmethod, ABC

import torch

from src.data_generation.streaming.beam_search.models.sampled_token import SampledToken


class BaseTokenSampler(ABC):
    @abstractmethod
    def sample_tokens(self, logits: torch.Tensor, token_logprob: torch.Tensor) -> SampledToken:
        pass
