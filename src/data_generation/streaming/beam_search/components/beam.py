from enum import Enum
from typing import Tuple, Union, Optional, List

import torch
from transformers import StoppingCriteriaList, PreTrainedModel

from src.data_processing.pipelines.abstract_pipeline import AbstractPipeline
from src.data_generation.streaming.beam_search.models.beam_output import BeamOutput
from src.data_generation.streaming.beam_search.models.sampled_token import SampledToken
from src.data_generation.streaming.stream_generator_config import StreamGeneratorConfig
from src.data_generation.streaming.beam_search.models.token_probs import TokenProbsResult
from src.data_generation.streaming.common.samplers.base import BaseTokenSampler


class StoppingReason(Enum):
    LOW_PROB_DIFF = "Low probability difference"
    STOP_CRITERIA = "StopCriteria"
    EOS = "EOS"
    ERROR = "Error"


class Beam:
    _instance_id = 0

    def __init__(
        self,
        model: PreTrainedModel,
        config: StreamGeneratorConfig,
        sampler: BaseTokenSampler,
        logits_processor_pipe: AbstractPipeline = None,
        stopping_criteria: StoppingCriteriaList = None
    ):
        self.model = model
        self.sampler = sampler

        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.unk_token_id = config.unk_token_id

        self.prob_threshold = config.prob_threshold
        self.length_penalty = config.length_penalty
        self.k_beam = config.k_beam

        self._beam_outputs = []
        self._next_past_key_values: Optional[Tuple] = None
        self._is_done = False
        self._id = Beam._instance_id

        self.logits_processor_pipe = logits_processor_pipe
        self.stopping_criteria = stopping_criteria

        Beam._instance_id += 1

    @property
    def next_past_key_values(self) -> Tuple:
        raise NotImplementedError("next_past_key_values is write only!")

    @next_past_key_values.setter
    def next_past_key_values(self, value: Tuple):
        self._next_past_key_values = value

    @property
    def outputs(self) -> List[BeamOutput]:
        return self._beam_outputs

    @property
    def id(self) -> int:
        return self._id

    @property
    def is_done(self):
        return self._is_done

    def __call__(self, input_ids: Union[torch.Tensor, torch.LongTensor, torch.FloatTensor]) -> BeamOutput:

        if self._is_done:
            raise ValueError(f"Beam[{self._id}] is already done!")

        token_probs_result, past_key_values = self.forward(input_ids)
        self._next_past_key_values = past_key_values

        reason = self.check_stopping_criteria(token_probs_result, input_ids)

        if reason is not None:
            self._is_done = True

        output = BeamOutput(token_probs_result, reason, past_key_values)
        self._beam_outputs.append(output)

        return output

    def forward(self, input_ids: torch.Tensor) -> Tuple[TokenProbsResult, Tuple]:

        inputs = self.model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=self._next_past_key_values,
            use_cache=True
        )

        with torch.inference_mode():
            output = self.model(
                **inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=True
            )

        logits: torch.Tensor = output.logits[:, -1, :]

        sequence_length = input_ids.size(1)
        length_penalty_factor = (5 + sequence_length) ** self.length_penalty / (5 + 1) ** self.length_penalty

        with torch.inference_mode():
            logits /= length_penalty_factor

        if self.logits_processor_pipe is not None:
            with torch.inference_mode():
                logits = self.logits_processor_pipe.apply(input_ids, logits)

        top_k_probs, top_k_token_ids = torch.topk(logits, self.k_beam, dim=-1)
        best_prob = torch.max(top_k_probs).item()

        # Use the _sample function to get sampled tokens and probabilities
        with torch.inference_mode():
            sampled_output: SampledToken = self.sampler.sample_tokens(logits, best_prob)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        past_key_values = output.past_key_values

        token_probs = TokenProbsResult(probs, sampled_output, top_k_probs)

        return token_probs, past_key_values

    def check_stopping_criteria(self, token_probs_result: TokenProbsResult, input_ids: torch.Tensor) -> Optional[str]:
        probs, sampled_token, top_k_probs = token_probs_result
        sampled_token_id, adjusted_prob = sampled_token

        # Custom stopping criteria based on probability difference
        if self.prob_threshold and (top_k_probs[0][0] - top_k_probs[0][1]).item() < self.prob_threshold:
            return StoppingReason.LOW_PROB_DIFF.value

        if self.stopping_criteria is not None and self.stopping_criteria(input_ids, probs):
            return StoppingReason.STOP_CRITERIA.value
        if sampled_token_id == self.eos_token_id:
            return StoppingReason.EOS.value
        elif sampled_token_id in [self.pad_token_id, self.unk_token_id]:
            return StoppingReason.ERROR.value

        return None
