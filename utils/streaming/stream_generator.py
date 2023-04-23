"""
A text generation model with stream decoding.
"""
from typing import List, Optional

import torch
from transformers import (
    LogitsProcessorList,
    MinNewTokensLengthLogitsProcessor,
    TemperatureLogitsWarper,
    NoRepeatNGramLogitsProcessor, RepetitionPenaltyLogitsProcessor, PreTrainedTokenizer, PreTrainedModel,
)

from utils.pipelines.abstract_pipeline import AbstractPipeline
from utils.pipelines.output_pipeline import OutputLogitsPipeline
from utils.streaming.helpers.choice_manager import ChoiceManager
from utils.streaming.helpers.models.stream_generator_config import StreamGeneratorConfig
from utils.streaming.helpers.models.stream_generator_kwargs import StreamGeneratorKwargs
from utils.streaming.helpers.stream_tokenizer import StreamTokenizer
from utils.pipelines.processors.flexible_top_p_logits import FlexibleTopPLogitsWarper
from utils.pipelines.processors.topic_bias_logits import TopicBiasLogitsProcessor


class StreamGenerator:
    """StreamGenerator wraps around a language model to provide stream decoding."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: StreamGeneratorConfig):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finish_reasons = None
        self.detokenizers = None
        self.config = config

    def __call__(
            self,
            prompt,
            min_tokens=0,
            max_tokens=32,
            n=1,
            logprobs=0,
            buffer_size=64,
            pipelines: List[AbstractPipeline] = None,
            echo=False,
    ):
        """Create a completion stream for the provided prompt."""
        self.reset()
        input_ids = self._process_prompt(prompt)
        min_tokens, max_tokens, n, logprobs = self._validate_arguments(min_tokens, max_tokens, n, logprobs)

        self.finish_reasons = [None] * n
        self.detokenizers = [StreamTokenizer(self.tokenizer) for _ in range(n)]

        if echo:
            self._echo_prompt(input_ids, self.detokenizers, logprobs, n)

        for output in self._generate_completion_tokens(input_ids, self.detokenizers, self.finish_reasons, min_tokens, max_tokens,
                                                       n, logprobs, buffer_size, pipelines):
            yield output

    def _process_prompt(self, prompt):
        if isinstance(prompt, str):
            input_ids = self.tokenize(prompt)
        elif isinstance(prompt, torch.Tensor) and prompt.dim() == 1:
            input_ids = prompt
        else:
            raise TypeError("prompt must be a string or a 1-d tensor")
        return input_ids

    @staticmethod
    def _validate_arguments(min_tokens, max_tokens, n, logprobs):
        min_tokens = max(min_tokens, 0)
        max_tokens = max(max_tokens, 1)
        n = max(n, 1)
        logprobs = max(logprobs, 0)
        return min_tokens, max_tokens, n, logprobs

    def _echo_prompt(self, input_ids, detokenizers, logprobs, n):
        for token in input_ids:
            samples = self._sample(token, 0, [], []) if logprobs > 0 else {}
            for i in range(n):
                text = detokenizers[i].decode(token)
                offset = detokenizers[i].start
                yield ChoiceManager.map_choice(text, i, text_offset=offset, **samples)

    def _generate_completion_tokens(self, input_ids, detokenizers, finish_reasons, min_tokens, max_tokens, n, logprobs, buffer_size, pipelines=None):
        token_buffers = [[] for _ in range(n)]

        for (masked_logits, output, status) in self._generate_tokens(input_ids, n, pipelines):
            tokens = output.sequences
            token_logprobs = output.sequences_log_probs
            top_tokens = output.top_k_ids
            top_logprobs = output.top_k_log_probs

            for i, (token, finished) in enumerate(zip(tokens, status)):
                if not finished and finish_reasons[i] is None:
                    samples = self._sample(token, token_logprobs[i], top_tokens[i],
                                           top_logprobs[i]) if logprobs > 0 else {}
                    text = detokenizers[i].decode(token)
                    offset = detokenizers[i].start

                    if text:
                        token_buffers[i].append(ChoiceManager.map_choice(text, i, text_offset=offset, **samples))

                    if detokenizers[i].has_reached_eos():
                        finish_reasons[i] = "eos"
                    elif detokenizers[i].length >= max_tokens:
                        finish_reasons[i] = "length"

                    if len(token_buffers[i]) >= buffer_size:
                        yield token_buffers[i]
                        token_buffers[i] = []

        # Yield the remaining tokens in the buffers
        for i in range(n):
            if token_buffers[i]:
                yield token_buffers[i]

    def _generate_tokens(self, input_ids, n, pipelines: Optional[List[AbstractPipeline]]=None):
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.bos_token_id = self.tokenizer.bos_token_id
        self.config.eos_token_id = self.tokenizer.eos_token_id
        self.config.decoder_start_token_id = self.tokenizer.decoder_start_token_id

        input_ids = input_ids.to(self.device)

        kwargs = StreamGeneratorKwargs(self.config, input_ids=input_ids, n=n)
        if pipelines is not None:
            for pipeline in pipelines:
                kwargs = pipeline.apply(self.config, kwargs)

        output: torch.Tensor = self.model.generate(**kwargs)

        # After obtaining the output from self.model.generate
        eos_token_id = self.tokenizer.eos_token_id
        status = (output != eos_token_id).all(dim=1).float()

        logits = output.logits
        masked_logits = logits * status.unsqueeze(-1)

        return masked_logits, output, status

    def reset(self):
        self.finish_reasons = [None] * self.n
        self.detokenizers = [StreamTokenizer(self.tokenizer) for _ in range(self.n)]

    def _sample(self, token, token_logprob, top_tokens, top_logprobs):
        samples = {
            "token": token,
            "token_logprob": token_logprob,
        }
        if top_tokens:
            samples.update(
                {"top_tokens": top_tokens, "top_logprobs": top_logprobs}
            )
        return samples

    def tokenize(self, text):
        return self.tokenizer.encode(text, return_tensors="pt")[0]
