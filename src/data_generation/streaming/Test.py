from typing import Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteriaList

from src.data_processing.pipelines.abstract_pipeline import AbstractPipeline
from src.data_processing.pipelines import OutputLogitsPipeline
from src.data_generation.streaming.stream_generator_config import StreamGeneratorConfig
from src.data_generation.streaming.common.samplers.base import BaseTokenSampler
from src.data_generation.streaming.stream_tokenizer import StreamTokenizer
from torch.nn import functional as F


class StreamGenerator:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, sampler: BaseTokenSampler,
                 config: StreamGeneratorConfig = None,
                 logits_processor_pipe: AbstractPipeline = None):

        if config is None:
            config = StreamGeneratorConfig()

        self.model = model
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.beam_size = config.beam_size
        self.chunk_size = config.chunk_size
        self.k_beam = config.k_beam
        self.logits_processor = logits_processor_pipe
        self.prob_threshold = config.prob_threshold
        self.length_penalty = config.length_penalty
        self.sampler = sampler

        # Create a separate StreamTokenizer for each beam
        self.stream_tokenizers = [StreamTokenizer(tokenizer) for _ in range(self.beam_size)]

    def _coherence_score(self, context, token):
        # Get the embeddings for the context and token
        context_embeddings = self._get_embeddings(context)
        token_embeddings = self._get_embeddings(token)

        # Compute the cosine similarity
        coherence_score = F.cosine_similarity(context_embeddings, token_embeddings).item()
        return coherence_score

    def __call__(self, prompt: str, stopping_criteria: StoppingCriteriaList = None):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = self._prepare_tensors(torch.device("cuda"), self.model, input_ids)

        self._adjust_default_log_processor(input_ids)

        yield from self.generate(input_ids[0], prompt, stopping_criteria)

    def generate(self, input_ids: torch.Tensor, prompt: str = None, stopping_criteria: StoppingCriteriaList = None):

        next_past_key_values = [None for _ in range(self.beam_size)]
        input_ids_list = [input_ids.clone() for _ in range(self.beam_size)]

        while True:
            beam_outputs = []
            stop_reasons = []
            past_key_values = []

            for i in range(self.beam_size):
                beam_output, adjusted_prob, next_past_key_value, reason = self._beam_forward(input_ids_list[i], next_past_key_values[i], None)
                decoded_output, original_text = self.stream_tokenizers[i].decode(token=beam_output)
                beam_outputs.append((beam_output.unsqueeze(0), adjusted_prob, next_past_key_value, reason, decoded_output))
                past_key_values.append(next_past_key_value)
                stop_reasons.append(reason)

            max_beam_output = max(beam_outputs, key=lambda x: x[1])
            decoded_text = max_beam_output[-1]

            for tokenizer in self.stream_tokenizers:
                tokenizer.add_to_history(decoded_text)

            yield decoded_text

            # Update the input_ids for each beam
            input_ids_list = [torch.cat([input_ids_list[i], max_beam_output[0]], dim=-1) for i in range(self.beam_size)]

            if input_ids_list[0].size(1) > self.max_length:
                input_ids_list = [input_ids[:, -self.max_length:] for input_ids in input_ids_list]

            next_past_key_values = [max_beam_output[2] for _ in range(self.beam_size)]

    def _beam_forward(
            self,
            input_ids: torch.Tensor,
            past_key_values: Tuple,
            stopping_criteria: StoppingCriteriaList = None
    ):
        reason = None
        inputs = self.model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, use_cache=True
        )

        with torch.inference_mode():
            output = self.model(
                **inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=True
            )

        logits = output.logits[:, -1, :]

        sequence_length = input_ids.size(1)
        length_penalty_factor = (5 + sequence_length) ** self.length_penalty / (5 + 1) ** self.length_penalty

        with torch.inference_mode():
            logits /= length_penalty_factor

        if self.logits_processor:
            with torch.inference_mode():
                logits = self.logits_processor.apply(input_ids, logits)

        top_k_probs, top_k_token_ids = torch.topk(logits, self.k_beam, dim=-1)
        best_prob = torch.max(top_k_probs).item()

        # Use the _sample function to get sampled tokens and probabilities
        with torch.inference_mode():
            sampled_output = self.sampler.sample_tokens(logits, best_prob)
            sampled_token_id = sampled_output["token"]
            adjusted_prob = sampled_output["token_logprob"]

        probs = torch.nn.functional.softmax(logits, dim=-1)
        past_key_values = output.past_key_values

        # Custom stopping criteria based on probability difference
        # Token selection based on a custom threshold
        if self.prob_threshold and (top_k_probs[0][0] - top_k_probs[0][1]).item() < self.prob_threshold:
            reason = "Low probability difference"

        # Use the StreamTokenizer for each beam to decode tokens
        decoded_token = self.tokenizer.decode(sampled_token_id)

        if stopping_criteria is not None and stopping_criteria(input_ids, probs):
            reason = "StopCriteria"

        # Check for stopping criteria
        if sampled_token_id == self.tokenizer.eos_token_id:
            reason = "EOS"
        elif len(decoded_token) == 0:
            reason = "Error"

        return sampled_token_id, adjusted_prob, past_key_values, reason

    def _adjust_default_log_processor(self, input_ids):
        if type(self.logits_processor) is OutputLogitsPipeline:
            # min_new_tokens_processor = self.logits_processor.processors[0]
            # min_new_tokens_processor.prompt_length_to_skip = input_ids[0].shape[-1]
            pass

    def _prepare_tensors(self, device: str, model, *tensors):
        tensors_d = []
        for tensor in tensors:
            if tensor is not None:
                tensors_d.append(tensor.to(device))
        self.model = model.to(device)
        return tuple(tensors_d)