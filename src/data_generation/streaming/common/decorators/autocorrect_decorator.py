from typing import List, Tuple

import torch
from transformers import AutoTokenizer, DistilBertForMaskedLM, PreTrainedTokenizer, BatchEncoding, DistilBertTokenizer, \
    DistilBertTokenizerFast, DistilBertConfig

from src.data_generation.streaming.beam_search.models.beam_output import BeamOutput
from src.data_generation.streaming.beam_search.models.beam_output_data import BeamOutputData
from src.data_generation.streaming.beam_search.models.sampled_token import SampledToken
from src.data_generation.streaming.common.decorators.abstract_decorator import AbstractDecorator
from src.data_generation.streaming.common.samplers.base import BaseTokenSampler
from src.data_generation.streaming.common.samplers.hybrid import HybridTokenSampler
from src.data_generation.streaming.stream_generator_config import StreamGeneratorConfig


class AutocorrectDecorator(AbstractDecorator):
    def __init__(self, config: StreamGeneratorConfig):
        model_name: str = "unnu10/distilroberta-base-finetuned-wikitext2"
        self.tokenizer: DistilBertTokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
        self.model: DistilBertForMaskedLM = DistilBertForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.sampler: HybridTokenSampler = HybridTokenSampler(self.tokenizer, config)

    def _correct_token(self, text: str) -> Tuple[str, torch.Tensor]:
        output_tokens: List[str] = self.tokenizer.tokenize(text)
        output_tokens[-1] = self.tokenizer.mask_token
        inputs: BatchEncoding = self.tokenizer(output_tokens, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        last_token_logits = logits[0, -1, :]

        with torch.inference_mode():
            sampled_token_data: SampledToken = self.sampler.sample_tokens(last_token_logits, token_k_logprob=None)

        sampled_token_id: int
        sampled_token_logprob: torch.Tensor
        sampled_token_id, sampled_token_logprob = sampled_token_data
        sampled_token_text: str = self.tokenizer.decode(sampled_token_id, skip_special_tokens=True)

        return sampled_token_text, sampled_token_logprob

    def process(self, beam_output_data: BeamOutputData, sequence: str) -> BeamOutputData:
        new_sequence: str = sequence + beam_output_data.decoded_output
        sequence_string_tokens: List[str] = self.tokenizer.tokenize(new_sequence)

        if self._is_empty_sequence(sequence_string_tokens) or self._is_short_sequence(sequence_string_tokens):
            return beam_output_data

        sequence_string_tokens[-1] = self.tokenizer.mask_token

        text: str = self.tokenizer.convert_tokens_to_string(sequence_string_tokens)

        self._correct_token(text)

        return BeamOutputData(BeamOutput(None, str(), tuple()), "")

    def _is_empty_sequence(self, sequence: List[str]) -> bool:
        special_tokens: List[str] = [
            self.tokenizer.pad_token,
            self.tokenizer.unk_token
        ]

        if self.tokenizer.mask_token is not None:
            special_tokens.append(self.tokenizer.mask_token)

        if self.tokenizer.eos_token is not None:
            special_tokens.append(self.tokenizer.eos_token)

        if self.tokenizer.bos_token is not None:
            special_tokens.append(self.tokenizer.bos_token)

        return len(sequence) == 0 or (len(sequence) == 1 and sequence[0] in special_tokens)

    def _is_short_sequence(self, sequence: List[str]) -> bool:
        return len(sequence) < 5

    def _normalize_text_tokens(self, token_list):
        text = ""
        for token in token_list:
            if token.startswith("##"):
                text += token[2:]
            else:
                if text:
                    text += " "
                text += token
        return text
