from abc import ABC, abstractmethod

from utils.streaming.helpers.models.stream_generator_config import StreamGeneratorConfig
from utils.streaming.helpers.models.stream_generator_kwargs import StreamGeneratorKwargs


class AbstractPipeline(ABC):
    @abstractmethod
    def apply(self, src_config: StreamGeneratorConfig, tgt_args: StreamGeneratorKwargs) -> StreamGeneratorKwargs:
        raise NotImplementedError
