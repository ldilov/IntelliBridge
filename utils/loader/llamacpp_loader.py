from pathlib import Path

from utils.loader.abstract_loader import AbstractLoader
from utils.streaming.response_stream import ResponseStream


class LlamaCppLoader(AbstractLoader):
    def __init__(self, name=None, dir=None):
        super().__init__(name)
        self.name = name
        self.dir = dir

    def load(self):
        from utils.llama.llamacpp_model import LlamaCppModel

        model_file = Path(f'{self.dir}/{self.name}')
        logger.warning(f"llama.cpp weights detected: {model_file}\n")

        model, tokenizer = LlamaCppModel.from_pretrained(model_file, ResponseStream)

        return model, tokenizer