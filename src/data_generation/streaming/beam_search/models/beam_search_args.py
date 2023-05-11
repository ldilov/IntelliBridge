from typing import Optional

from transformers import StoppingCriteriaList, PreTrainedModel, PreTrainedTokenizer

from src.data_processing.pipelines.abstract_pipeline import AbstractPipeline
from src.data_generation.streaming.stream_generator_config import StreamGeneratorConfig
from src.data_generation.streaming.common.samplers.base import BaseTokenSampler


class BeamSearchArgs:
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            sampler: BaseTokenSampler,
            config: Optional[StreamGeneratorConfig],
            logits_processor_pipe: Optional[AbstractPipeline] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_length
        self.beam_size = config.beam_size
        self.k_beam = config.k_beam
        self.prob_threshold = config.prob_threshold
        self.length_penalty = config.length_penalty
        self.sampler = sampler

        self.stopping_criteria: Optional[StoppingCriteriaList] = stopping_criteria
        self.logits_processor_pipe: Optional[AbstractPipeline] = logits_processor_pipe

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in vars(self)}
