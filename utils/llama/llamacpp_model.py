import inspect
import multiprocessing

from llama_cpp import Llama, LlamaCache

from utils.streaming.response_stream import ResponseStream


class LlamaCppModel:
    def __init__(self):
        self.initialized = False
        self.model = None
        self.response_stream_cls = None
        self.eos_token_id = None
        self.bos_token_id = None
        self.pad_token_id = None

    @classmethod
    def from_pretrained(cls, path, n_threads=None):
        result = cls()

        params = {
            'model_path': str(path),
            'n_ctx': 2048,
            'embedding': True,
            'seed': 0,
            'n_batch': 8,
            'n_threads': n_threads or int(multiprocessing.cpu_count() * 0.75)
        }

        result.model = Llama(**params)
        result.model.set_cache(LlamaCache)

        result.eos_token_id = 2
        result.bos_token_id = 1
        result.unk_token_id = 0
        result.pad_token_id = -1

        return result, result

    def tokenize(self, string):
        return self.encode(string)

    def encode(self, string):
        if type(string) is str:
            string = string.encode()
        return self.model.tokenize(string)

    def decode(self, *args, **kwargs):
        return self._decode_inner(args[0])

    def generate(self, input="", token_count=20, temperature=1, top_p=1, top_k=50, repetition_penalty=1, callback=None):
        if type(input) is str:
            input = input.encode()
        tokens = self.model.tokenize(input)

        if type(top_p) is float:
            top_p = int(top_p * 100)

        if type(top_k) is float:
            top_k = int(top_k * 100)

        output = b""
        count = 0
        for token in self.model.generate(tokens, top_k=top_k, top_p=top_p, temp=temperature, repeat_penalty=repetition_penalty):
            text = self.model.detokenize([token])
            output += text

            if callback:
                callback(text.decode('utf-8'))

            count += 1
            if count >= token_count or (token == self.model.token_eos()):
                break

        result = str(output.decode('utf-8'))

        return result

    def generate_with_streaming(self, **kwargs):
        with self.response_stream_cls(self.generate, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply

    def _decode_inner(self, tokens):
        return self.model.detokenize(tokens)