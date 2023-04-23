import ast
import traceback

import torch
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer

from kernel.logger.logger import logger
from kernel.persistence.infra.models.abstract_model import AbstractModel
from utils.stopping_criteria.sentinel_token_stopping_criteria import SentinelTokenStoppingCriteria
from utils.stopping_criteria.special_tokens_stopping_criteria import SpecialTokensStoppingCriteria
from utils.streaming.response_stream import ResponseStream
from utils.streaming.stream import Stream


class BasicGenerator(object):
    @classmethod
    def generate(cls, model: AbstractModel, input_text: str, state_params: dict):
        tokenizer: PreTrainedTokenizer = model.tokenizer
        eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

        # Handling the stopping strings
        stopping_criteria_list = transformers.StoppingCriteriaList()
        stopping_strings = []

        # for st in (stopping_strings, ast.literal_eval(f"[{state_params['custom_stopping_strings']}]")):
        #     if type(st) is list and len(st) > 0:
        #         sentinel_token_ids = [tokenizer.encode(string, add_special_tokens=False) for string in st]
        #
        #         sentinel = SentinelTokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids,
        #                                                  starting_idx=len(input_ids[0]))
        #         spec_tokens = SpecialTokensStoppingCriteria(tokenizer, starting_idx=len(input_ids[0]))
        #         stopping_criteria_list.append(sentinel)
        #         stopping_criteria_list.append(spec_tokens)
        #         break

        generate_params = {}
        for k in ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty',
                  'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams',
                  'penalty_alpha', 'length_penalty', 'early_stopping']:
            generate_params[k] = state_params[k]
        generate_params['eos_token_id'] = eos_token_ids
        generate_params['stopping_criteria'] = stopping_criteria_list
        if state_params['ban_eos_token']:
            generate_params['suppress_tokens'] = [tokenizer.eos_token_id]

        from kernel.persistence.memory.global_registry import registry

        args = registry.get('args')

        if args.no_cache:
            generate_params.update({'use_cache': False})
        if args.deepspeed:
            generate_params.update({'synced_gpus': True})

        try:
            def generate_with_streaming(**kwargs):
                with torch.no_grad():
                    return model.generate(**kwargs)

            for output in generate_with_streaming(input_text=input_text):
                reply = tokenizer.decode(output, skip_special_tokens=True)
                if output[-1] in eos_token_ids:
                    break

                # yield gen_class.formatted_outputs(reply, gen_class.args.model)

        except Exception:
            logger.exception("Error in generating text response!")
