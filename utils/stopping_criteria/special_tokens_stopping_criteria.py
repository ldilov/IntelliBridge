import torch
import transformers

from utils.llama.llamacpp_tokenizer import LlamaCppTokenizer


class SpecialTokensStoppingCriteria(transformers.StoppingCriteria):
    DEFAULT_SPECIAL_TOKENS = ["[end of text]", "<|endoftext|>", "???"]

    def __init__(self, tokenizer, starting_idx: int, special_tokens=None):
        transformers.StoppingCriteria.__init__(self)

        if special_tokens is None:
            special_tokens = []

        tokens = SpecialTokensStoppingCriteria.DEFAULT_SPECIAL_TOKENS

        if special_tokens is not None:
            tokens = tokens + special_tokens
            tokens = list(set(tokens))

        self.special_tokens = [tokenizer.encode(token, add_special_tokens=False) for token in tokens]
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]

            for i in range(len(self.special_tokens)):
                if trimmed_sample.shape[-1] < self.special_tokens[i].shape[-1]:
                    continue
                for window in trimmed_sample.unfold(0, self.special_tokens[i].shape[-1], 1):
                    if torch.all(torch.eq(self.special_tokens[i][0], window)):
                        return True
        return False
