import multiprocessing

from llamacpp import llamacpp

from src.data_generation.llama.llamacpp_tokenizer import LlamaCppTokenizer
from src.data_generation.streaming import ResponseStream


class LlamaCppModelNative:
    def __init__(self):
        self.initialized = False
        self.model = None
        self.params = None
        self.response_stream_cls = None

    @classmethod
    def from_pretrained(
            self,
            path,
            tokenizer_cls: LlamaCppTokenizer,
            response_stream_cls: ResponseStream,
            n_threads=1
    ):
        params = llamacpp.InferenceParams()
        params.path_model = str(path)
        params.n_threads = n_threads or int(multiprocessing.cpu_count() * 0.75)

        _model = llamacpp.LlamaInference(params)

        result = self()
        result.model = _model
        result.params = params
        result.response_stream_cls = response_stream_cls

        tokenizer = tokenizer_cls.from_model(_model)
        return result, tokenizer

    def generate(self, context="", token_count=20, temperature=1, top_p=1, top_k=50, repetition_penalty=1,
                 callback=None):
        params = self.params
        params.n_predict = token_count
        params.top_p = top_p
        params.top_k = top_k
        params.temp = temperature
        params.repeat_penalty = repetition_penalty
        # params.repeat_last_n = repeat_last_n

        # self.model.params = params
        self.model.add_bos()
        self.model.update_input(context)

        output = ""
        is_end_of_text = False
        ctr = 0
        while ctr < token_count and not is_end_of_text:
            if self.model.has_unconsumed_input():
                self.model.ingest_all_pending_input()
            else:
                self.model.eval()
                token = self.model.sample()
                text = self.model.token_to_str(token)
                output += text
                is_end_of_text = token == self.model.token_eos()
                if callback:
                    callback(text)
                ctr += 1

        return output

    def generate_with_streaming(self, **kwargs):
        with self.response_stream_cls(self.generate, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
