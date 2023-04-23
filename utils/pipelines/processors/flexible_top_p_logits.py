import torch
from transformers import LogitsProcessor, TopPLogitsWarper


class FlexibleTopPLogitsWarper(LogitsProcessor):
    def __init__(self, min_top_p=0.1, max_top_p=0.9, transition_length=50, scale=1.0):
        self.min_top_p = min_top_p
        self.max_top_p = max_top_p
        self.transition_length = transition_length
        self.scale = scale

        assert 0.0 <= self.min_top_p <= 1.0, "min_top_p should be between 0 and 1"
        assert 0.0 <= self.max_top_p <= 1.0, "max_top_p should be between 0 and 1"
        assert self.transition_length > 0, "transition_length should be greater than 0"

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x * self.scale))

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[1]
        t = (cur_len - self.transition_length) / self.transition_length
        top_p = self.min_top_p + (self.max_top_p - self.min_top_p) * (1 - self.sigmoid(t))

        return TopPLogitsWarper(top_p=top_p)(input_ids, scores)