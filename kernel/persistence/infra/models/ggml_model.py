from typing import TypeVar, Generic, Type

from kernel.persistence.infra.models.abstract_model import AbstractModel

T = TypeVar('T')


class GgmlModel(Generic[T], AbstractModel):
    def __init__(self, model, tokenizer, response_type: Type[T]):
        super().__init__(model, tokenizer)
        self.model.response_stream_cls = response_type

    def encode(self, string, *args, **kwargs):
        if type(string) is str:
            string = string.encode()
        return self.tokenizer.tokenize(string, *args, **kwargs)

    def decode(self, input_ids, *args, **kwargs):
        return self.tokenizer.decode(input_ids, *args, **kwargs)

    def generate(self, *args, **kwargs):
        yield self.model.generate_with_streaming(*args, ** kwargs)
