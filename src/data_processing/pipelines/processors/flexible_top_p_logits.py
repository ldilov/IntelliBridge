import torch
from transformers import LogitsProcessor, TopPLogitsWarper

class FlexibleTopPLogitsWarper(LogitsProcessor):
    def __init__(self, min_top_p=0.1, max_top_p=0.9, transition_length=50, scale=1.0):
        self.min_top_p = min_top_p
        self.max_top_p = max_top_p
        self.transition_length = transition_length
        self.scale = scale
        self.generated_len = 0

        assert 0.0 <= self.min_top_p <= 1.0, "min_top_p should be between 0 and 1"
        assert 0.0 <= self.max_top_p <= 1.0, "max_top_p should be between 0 and 1.0"
        assert self.transition_length > 0, "transition_length should be greater than 0"

        self.warpers = {}

    def sigmoid(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float)  # Convert x to a tensor
        return 1 / (1 + torch.exp(-x_tensor * self.scale))

    def __call__(self, input_ids, scores):
        self.generated_len += 1

        # Get the appropriate TopPLogitsWarper for the current input_ids tensor
        if input_ids not in self.warpers:
            # Create a new TopPLogitsWarper for this input_ids tensor
            self.warpers[input_ids] = TopPLogitsWarper(
                top_p=self.min_top_p,
                filter_value=-float("Inf"),
                min_tokens_to_keep=2
            )
        warper = self.warpers[input_ids]

        if self.generated_len < self.transition_length:
            top_p = self.min_top_p
        else:
            t = (self.generated_len - self.transition_length) / self.transition_length
            top_p = self.min_top_p + (self.max_top_p - self.min_top_p) * (1 - self.sigmoid(t))

        # Update the top_p value for the TopPLogitsWarper
        warper.top_p = top_p

        return warper(input_ids, scores)
