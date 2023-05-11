import torch

from src.data_generation.streaming.stream_generator_config import StreamGeneratorConfig
from src.data_generation.streaming.beam_search.models.sampled_token import SampledToken
from src.data_generation.streaming.common.samplers.base import BaseTokenSampler


class HybridTokenSampler(BaseTokenSampler):

    def __init__(self, tokenizer, config: StreamGeneratorConfig):
        self.tokenizer = tokenizer
        self.temperature = config.temperature
        self.top_k = config.top_k

    def sample_tokens(self, logits: torch.Tensor, token_k_logprob) -> SampledToken:
        """Sample log probabilities of the most likely tokens using top-k sampling."""

        epsilon = 1e-9

        # Apply temperature
        logits += epsilon
        logits /= self.temperature

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Top-k sampling
        top_k_probs, top_k_token_ids = torch.topk(probs, self.top_k, dim=-1)

        n_sample = min(1, top_k_probs.size(-1))
        sampled_token_index = torch.multinomial(top_k_probs, n_sample).squeeze(0)
        sampled_token_id = top_k_token_ids[0, sampled_token_index]
        token_k_logprob = torch.log(top_k_probs[0, sampled_token_index])

        sampled_token_data = SampledToken(sampled_token_id, token_k_logprob.item())

        return sampled_token_data
