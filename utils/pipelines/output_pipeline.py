from transformers import GenerationConfig, TemperatureLogitsWarper, LogitsProcessorList, \
    MinNewTokensLengthLogitsProcessor, NoRepeatNGramLogitsProcessor, RepetitionPenaltyLogitsProcessor

from utils.pipelines.abstract_pipeline import AbstractPipeline
from utils.pipelines.processors.flexible_top_p_logits import FlexibleTopPLogitsWarper
from utils.pipelines.processors.topic_bias_logits import TopicBiasLogitsProcessor
from utils.streaming.helpers.models.stream_generator_config import StreamGeneratorConfig
from utils.streaming.helpers.models.stream_generator_kwargs import StreamGeneratorKwargs


class OutputLogitsPipeline(AbstractPipeline):
    def __init__(self):
        super().__init__()

    def apply(self, src_config: StreamGeneratorConfig, tgt_args: StreamGeneratorKwargs) -> StreamGeneratorKwargs:
        processors = self.__create_logits_processors(src_config)
        tgt_args['logits_processor'] = processors
        return tgt_args

    def __create_logits_processors(self, config: StreamGeneratorConfig) -> LogitsProcessorList:
        processors = LogitsProcessorList([
            MinNewTokensLengthLogitsProcessor(config.min_new_tokens),
            TemperatureLogitsWarper(config.temperature),
            NoRepeatNGramLogitsProcessor(config.no_repeat_ngram_size),
            RepetitionPenaltyLogitsProcessor(config.repetition_penalty),
            TopicBiasLogitsProcessor(config.topic_bias_keywords, config.topic_bias_weight),
            FlexibleTopPLogitsWarper(config.min_top_p, config.max_top_p, config.top_p_transition_length,
                                     config.top_p_scale),

        ])

        return processors
