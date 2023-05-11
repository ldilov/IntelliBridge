import torch
from transformers import LogitsProcessor


class CreativeBoostLogitsProcessor(LogitsProcessor):
    def __init__(self, rare_token_ids, boost_value=0.1):
        self.rare_token_ids = rare_token_ids
        self.boost_value = boost_value

    def __call__(self, input_ids, scores):
        if not self.boost_value or \
                self.rare_token_ids is None or \
                len(self.rare_token_ids) == 0:
            mask = torch.zeros_like(scores)
            mask[:, self.rare_token_ids] = self.boost_value
            scores = scores + mask
            return scores
