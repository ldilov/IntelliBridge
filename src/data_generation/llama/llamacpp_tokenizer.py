import llamacpp


class LlamaCppTokenizer:
    """A thin wrapper over the llamacpp tokenizer"""
    def __init__(self, model: llamacpp.LlamaInference):
        self._tokenizer = model.get_tokenizer()
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.pad_token_id = -1

    @classmethod
    def from_model(cls, model: llamacpp.LlamaInference):
        return cls(model)

    def encode(self, prompt: str):
        return self._tokenizer.tokenize(prompt)

    def decode(self, ids):
        return self._tokenizer.detokenize(ids)