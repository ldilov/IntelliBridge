import torch
import transformers


class SentinelTokenStoppingCriteria(transformers.StoppingCriteria):
    """
    A stopping criteria that stops when a sentinel token is generated.
    :Ref https://github.com/mk-cupist
    """
    def __init__(self, sentinel_token_ids: list, starting_idx: int):
        transformers.StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]

            for i in range(len(self.sentinel_token_ids)):
                # Can't unfold, output is still too tiny. Skip.
                if trimmed_sample.shape[-1] < self.sentinel_token_ids[i].shape[-1]:
                    continue
                for window in trimmed_sample.unfold(0, self.sentinel_token_ids[i].shape[-1], 1):
                     if torch.all(torch.eq(self.sentinel_token_ids[i][0], window)):
                        return True
        return False
