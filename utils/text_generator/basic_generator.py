import ast
import traceback

import torch
import transformers

from kernel.logger.logger import logger
from utils.stopping_criteria.sentinel_token_stopping_criteria import SentinelTokenStoppingCriteria
from utils.stopping_criteria.special_tokens_stopping_criteria import SpecialTokensStoppingCriteria
from utils.streaming.response_stream import ResponseStream
from utils.streaming.stream import Stream


class BasicGenerator(object):
    @classmethod
    def generate(cls, gen_class, question, state_params, stopping_strings, eos_token, generate_params):
        input_ids = gen_class.encode(question, add_bos_token=state_params['add_bos_token'],
                                truncation_length=gen_class.get_max_prompt_length(state_params))

        eos_token_ids = [gen_class.tokenizer.eos_token_id] if gen_class.tokenizer.eos_token_id is not None else []
        if eos_token is not None:
            eos_token_ids.append(int(gen_class.encode(eos_token)[0][-1]))

        # Handling the stopping strings
        stopping_criteria_list = transformers.StoppingCriteriaList()
        for st in (stopping_strings, ast.literal_eval(f"[{state_params['custom_stopping_strings']}]")):
            if type(st) is list and len(st) > 0:
                sentinel_token_ids = [gen_class.encode(string, add_special_tokens=False) for string in st]

                sentinel = SentinelTokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=len(input_ids[0]))
                spec_tokens = SpecialTokensStoppingCriteria(gen_class.tokenizer, starting_idx=len(input_ids[0]))
                stopping_criteria_list.append(sentinel)
                stopping_criteria_list.append(spec_tokens)
                break

        for k in ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty',
                  'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams',
                  'penalty_alpha', 'length_penalty', 'early_stopping']:
            generate_params[k] = state_params[k]
        generate_params['eos_token_id'] = eos_token_ids
        generate_params['stopping_criteria'] = stopping_criteria_list
        if state_params['ban_eos_token']:
            generate_params['suppress_tokens'] = [gen_class.tokenizer.eos_token_id]

        if generate_params.args.no_cache:
            generate_params.update({'use_cache': False})
        if generate_params.args.deepspeed:
            generate_params.update({'synced_gpus': True})
        if generate_params.soft_prompt:
            inputs_embeds, filler_input_ids = generate_params.generate_softprompt_input_tensors(input_ids)
            generate_params.update({'inputs_embeds': inputs_embeds})
            generate_params.update({'inputs': filler_input_ids})
        else:
            generate_params.update({'inputs': input_ids})

        try:
            def generate_with_callback(callback=None, **kwargs):
                kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                with torch.no_grad():
                    generate_params.model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return ResponseStream(generate_with_callback, kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    if generate_params.soft_prompt:
                        output = torch.cat((input_ids[0], output[filler_input_ids.shape[1]:]))

                    print(list(output))
                    new_tokens = len(output) - len(input_ids[0])
                    reply = generate_params.decode(output[-new_tokens:], state_params['skip_special_tokens'])
                    if output[-1] in eos_token_ids:
                        break

                    yield gen_class.formatted_outputs(reply, gen_class.args.model)

        except Exception:
            logger.exception("Error in generating text response!")