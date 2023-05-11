import torch
from transformers import LogitsProcessorList, NoRepeatNGramLogitsProcessor
from src.data_processing.pipelines.abstract_pipeline import AbstractPipeline
from src.data_generation.streaming.stream_generator_config import StreamGeneratorConfig


class OutputLogitsPipeline(AbstractPipeline):
    def __init__(self, src_config: StreamGeneratorConfig):
        super().__init__()
        self.__src_config: StreamGeneratorConfig = src_config
        self.__processors: LogitsProcessorList = self.__create_logits_processors(self.__src_config)

    def apply(self, input_ids: torch.Tensor, logits):
        logits = self.__processors(input_ids, scores=logits)
        return logits

    def __create_logits_processors(self, config: StreamGeneratorConfig) -> LogitsProcessorList:
        processors = LogitsProcessorList([
            # MinNewTokensLengthLogitsProcessor(20, config.min_new_tokens, config.eos_token_id),
            # FlexibleTemperatureLogitsProcessor(config.min_temperature, config.max_temperature,
            #                                    config.temperature_transition_length),
            NoRepeatNGramLogitsProcessor(config.no_repeat_ngram_size),
            # RepetitionPenaltyLogitsProcessor(config.repetition_penalty),
            # TopicBiasLogitsProcessor(config.topic_bias_keywords, config.topic_bias_weight),
            # FlexibleTopPLogitsWarper(config.min_top_p, config.max_top_p, config.top_p_transition_length,
            #                          config.top_p_scale),

        ])

        return processors

    @property
    def processors(self) -> LogitsProcessorList:
        return self.__processors
