from typing import List, Iterator

import torch
from transformers import PreTrainedTokenizer

from src.data_generation.streaming.beam_search.models.beam_search_args import BeamSearchArgs
from src.data_generation.streaming.beam_search.components.beam import Beam
from src.data_generation.streaming.beam_search.models.beam_output import BeamOutput
from src.data_generation.streaming.beam_search.models.beam_output_data import BeamOutputData
from src.data_generation.streaming.common.decorators.autocorrect_decorator import AutocorrectDecorator
from src.data_generation.streaming.common.decorators.tokenizer_decode_decorator import TokenizerDecodeDecorator
from src.data_generation.streaming.common.strategies.base_strategy import DecodingStrategy
from src.data_generation.streaming.stream_decoder import StreamDecoder
from src.data_generation.streaming.stream_tokenizer import StreamTokenizer


class BeamSearchStrategy(DecodingStrategy):
    def __init__(self, args: BeamSearchArgs):
        super().__init__(
            args.model,
            args.config,
            args.sampler,
            args.stopping_criteria,
            args.logits_processor_pipe
        )

        self.tokenizer: PreTrainedTokenizer = args.tokenizer
        self.beam_size: int = args.config.beam_size
        self.stream_tokenizers: List[StreamTokenizer] = [StreamTokenizer(args.tokenizer) for _ in range(self.beam_size)]

        self.beam_args = (self._model, self._config, self._sampler, self._logits_processor_pipe, self._stopping_criteria)
        self.beams: List[Beam] = [Beam(*self.beam_args) for _ in range(self.beam_size)]

        self.prob_diff_threshold: float = args.config.prob_threshold
        self.beam_outputs: List[BeamOutputData] = []
        self.input_ids_list = []
        self.stop_condition = False

        self.decoders = [StreamDecoder([]) for _ in range(self.beam_size)]

    def generate(self, input_ids: torch.Tensor) -> Iterator[str]:
        self.input_ids_list = [input_ids.clone() for _ in range(self.beam_size)]

        while self.stop_condition is False:
            beam_outputs: List[BeamOutputData] = []

            for index in range(self.beam_size):
                beam: Beam = self.beams[index]

                if beam.is_done:
                    self.beams[index] = Beam(*self.beam_args)
                    beam = self.beams[index]

                beam_output_result: BeamOutput = beam(self.input_ids_list[index])
                beam_output_text: str = self.stream_tokenizers[index].decode(beam_output_result.token_id)
                beam_output_data: BeamOutputData = BeamOutputData(beam_output_result, beam_output_text)

                tokenizer_decode_decorator = TokenizerDecodeDecorator(self.stream_tokenizers[index])
                autocorrect_decorator = AutocorrectDecorator(self._config)
                decorators = [tokenizer_decode_decorator, autocorrect_decorator]
                self.decoders[index].extend_decorators(decorators)
                beam_output_data = self.decoders[index].process(beam_output_data)

                beam_outputs.append(beam_output_data)

            max_beam_output: BeamOutputData = max(beam_outputs)
            self._apply_stop_condition(max_beam_output)
            decoded_text = str(max_beam_output)

            for tokenizer in self.stream_tokenizers:
                tokenizer.add_to_history(decoded_text)

            yield decoded_text

            self._update_input_ids_and_state(max_beam_output)

        self._reset()

    def _reset(self):
        beam_args = (self._model, self._config, self._sampler, self._logits_processor_pipe, self._stopping_criteria)

        self.stop_condition = False
        self.beams: List[Beam] = [Beam(*beam_args) for _ in range(self.beam_size)]
        self.stream_tokenizers: List[StreamTokenizer] = [StreamTokenizer(self.tokenizer) for _ in range(self.beam_size)]

    def _apply_stop_condition(self, max_beam_output: BeamOutputData):
        top_k_probs = max_beam_output.top_k_probs
        if len(top_k_probs) > 1:
            prob_diff = top_k_probs[0][0] - top_k_probs[0][1]
            if prob_diff.item() < self.prob_diff_threshold:
               self.stop_condition = True

    def _update_input_ids_and_state(self, max_beam_output: BeamOutputData):
        output_id = max_beam_output.token_id.unsqueeze(0)
        self.input_ids_list = [torch.cat([self.input_ids_list[i], output_id], dim=-1) for i in range(self.beam_size)]

        if self.input_ids_list[0].size(1) > self._max_length:
            self.input_ids_list = [input_ids[:, -self._max_length:] for input_ids in self.input_ids_list]

        for beam in self.beams:
            beam: Beam
            beam.next_past_key_values = max_beam_output.past_key_values
