import torch
from transformers import LogitsProcessor


class TopicBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, keyword_ids, bias_value=2.0):
        self.keyword_ids = keyword_ids
        self.bias_value = bias_value

    def __call__(self, input_ids, scores):
        if self.bias_value == 0.0 or \
                self.keyword_ids is None or \
                len(self.keyword_ids) == 0:
            return scores

        mask = torch.zeros_like(scores)
        mask[:, self.keyword_ids] = self.bias_value
        scores = scores + mask
        return scores
