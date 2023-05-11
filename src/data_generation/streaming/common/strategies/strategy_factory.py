from typing import Optional

from transformers import PreTrainedModel, StoppingCriteriaList, PreTrainedTokenizer

from src.data_processing.pipelines.abstract_pipeline import AbstractPipeline
from src.data_generation.streaming.beam_search.models.beam_search_args import BeamSearchArgs
from src.data_generation.streaming.stream_generator_config import StreamGeneratorConfig
from src.data_generation.streaming.common.samplers.base import BaseTokenSampler
from src.data_generation.streaming.common.strategies.beam_search_strategy import BeamSearchStrategy


class StrategyFactory:
    @staticmethod
    def create_strategy(
            strategy_type: str,
            model: PreTrainedModel,
            config: StreamGeneratorConfig,
            tokenizer: PreTrainedTokenizer,
            sampler: BaseTokenSampler,
            logits_processor_pipe: Optional[AbstractPipeline] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            **specific_args
    ):
        if strategy_type == 'auto_regression':
            raise NotImplementedError("AutoRegressionStrategy is not implemented yet!")
        elif strategy_type == 'beam_search':
            args: BeamSearchArgs = BeamSearchArgs(
                model,
                tokenizer,
                sampler,
                config,
                logits_processor_pipe,
                stopping_criteria
            )
            return BeamSearchStrategy(args)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
