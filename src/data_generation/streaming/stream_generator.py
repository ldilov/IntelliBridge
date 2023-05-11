from typing import Optional, Iterator

from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteriaList

from src.data_processing.pipelines.abstract_pipeline import AbstractPipeline
from src.data_generation.streaming.stream_generator_config import StreamGeneratorConfig
from src.data_generation.streaming.common.samplers.base import BaseTokenSampler
from src.data_generation.streaming.common.strategies.base_strategy import DecodingStrategy
from src.data_generation.streaming.common.strategies.strategy_factory import StrategyFactory


class StreamGenerator:
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            sampler: BaseTokenSampler,
            config: Optional[StreamGeneratorConfig] = None,
            logits_processor_pipe: Optional[AbstractPipeline] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            strategy_type: Optional[str] = 'beam_search',
            device: Optional[str] = 'cuda',
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        if config is None:
            config = StreamGeneratorConfig()

        self.generation_strategy: DecodingStrategy = StrategyFactory.create_strategy(
            strategy_type=strategy_type,
            model=model,
            config=config,
            tokenizer=tokenizer,
            sampler=sampler,
            logits_processor_pipe=logits_processor_pipe,
            stopping_criteria=stopping_criteria,
        )

    def __call__(self, input_ids, **kwargs) -> Iterator[str]:
        yield from self.generate(input_ids)

    def generate(self, prompt: str) -> Iterator[str]:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = self._prepare_tensors(self.device, self.model, input_ids)
        yield from self.generation_strategy.generate(input_ids[0])

    def _prepare_tensors(self, device: str, model, *tensors):
        tensors_d = []
        for tensor in tensors:
            if tensor is not None:
                tensors_d.append(tensor.to(device))
        self.model = model.to(device)
        return tuple(tensors_d)
