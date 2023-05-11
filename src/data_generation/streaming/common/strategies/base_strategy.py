from abc import abstractmethod, ABC
from typing import Iterator, Optional

import torch
from transformers import StoppingCriteriaList, PreTrainedModel

from src.data_processing.pipelines.abstract_pipeline import AbstractPipeline
from src.data_generation.streaming.stream_generator_config import StreamGeneratorConfig
from src.data_generation.streaming.common.samplers.base import BaseTokenSampler


class DecodingStrategy(ABC):

    def __init__(
            self,
            model: PreTrainedModel,
            config: StreamGeneratorConfig,
            sampler: BaseTokenSampler,
            stopping_criteria: Optional[StoppingCriteriaList],
            logits_processor_pipe: Optional[AbstractPipeline],
    ):
        self._config = config
        self._model = model
        self._sampler = sampler
        self._max_length = config.max_length
        self._logits_processor_pipe = logits_processor_pipe
        self._stopping_criteria = stopping_criteria

    @abstractmethod
    def generate(self, input_ids: torch.Tensor) -> Iterator[str]:
        pass
