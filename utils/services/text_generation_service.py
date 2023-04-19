import ast
import inspect
import os
import random
import time
import traceback

import numpy as np
import torch
import transformers
from pathlib import Path

from utils.streaming.response_stream import ResponseStream
from utils.text_generator.basic_generator import BasicGenerator
from utils.text_generator.llamacpp_generator import LlamaCppGenerator
from utils.text_processor.galactica import Galactica
from utils.text_processor.gpt import GPT
from utils.text_processor.gpt4chan import Gpt4Chan
from utils.pytorch.gc import TorchGC


class TextGenerationService(object):
    def __init__(self, args, model, tokenizer, soft_prompt=None, soft_prompt_tensor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.soft_prompt = soft_prompt
        self.soft_prompt_tensor = soft_prompt_tensor
        self.is_llamacpp = "ggml" in Path(f'{args.model_dir}/{args.model}').name
        self.args = args
        self.gc = TorchGC(args)
        self.local_rank = self.args.local_rank if self.args.local_rank is not None else int(
            os.getenv("LOCAL_RANK", "0"))

    def get_max_prompt_length(self, state):
        max_length = state['truncation_length'] - state['max_new_tokens']
        if self.soft_prompt:
            max_length -= self.soft_prompt_tensor.shape[1]
        return max_length

    def encode(self, prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
        if self.is_llamacpp:
            input_ids = self.tokenizer.encode(str(prompt))
            input_ids = np.array(input_ids).reshape(1, len(input_ids))
            return input_ids
        else:
            input_ids = self.tokenizer.encode(str(prompt), return_tensors='pt', add_special_tokens=add_special_tokens)

            # This is a hack for making replies more creative.
            if not add_bos_token and input_ids[0][0] == self.tokenizer.bos_token_id:
                input_ids = input_ids[:, 1:]

            # Llama adds this extra token when the first character is '\n', and this
            # compromises the stopping criteria, so we just remove it
            if type(self.tokenizer) is transformers.LlamaTokenizer and input_ids[0][0] == 29871:
                input_ids = input_ids[:, 1:]

        # Handling truncation
        if truncation_length is not None:
            input_ids = input_ids[:, -truncation_length:]

        if any((self.is_llamacpp, self.args.cpu)):
            return input_ids
        elif self.args.deepspeed:
            return input_ids.to(device=self.local_rank)
        elif torch.has_mps:
            device = torch.device('mps')
            return input_ids.to(device)
        else:
            return input_ids.cuda()

    def decode(self, output_ids, skip_special_tokens=True):
        if skip_special_tokens:
            reply = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            reply = GPT.clean_response(reply)
            return reply
        else:
            return self.tokenizer.decode(output_ids, skip_special_tokens=False)

    def generate_softprompt_input_tensors(self, input_ids):
        inputs_embeds = self.model.transformer.wte(input_ids)
        inputs_embeds = torch.cat((self.soft_prompt_tensor, inputs_embeds), dim=1)
        filler_input_ids = torch.zeros((1, inputs_embeds.shape[1]), dtype=input_ids.dtype).to(self.model.device)

        return inputs_embeds, filler_input_ids

    def formatted_outputs(self, reply, model_name):
        if 'galactica' in model_name.lower():
            reply = Galactica.clean(reply)
            return reply, reply
        elif any((k in model_name.lower() for k in ['gpt4chan', 'gpt-4chan'])):
            reply = Gpt4Chan.clean(reply)
            return reply, 'Only applicable for GALACTICA models.'
        else:
            return reply, 'Only applicable for GALACTICA models.'

    def set_manual_seed(self, seed):
        seed = int(seed)
        if seed == -1:
            seed = random.randint(1, 2 ** 31)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    def stop_everything_event(self):
        ResponseStream.stop()

    def generate_reply(self, question, state_params, eos_token=None, stopping_strings=[]):
        if self.model is None:
            raise ModuleNotFoundError("No model is loaded! Select one in the Model tab.")
            return

        self.gc.clear_torch_cache()
        self.set_manual_seed(state_params['seed'])
        generate_params = {}

        if self.is_llamacpp:
            return LlamaCppGenerator.generate(
                self,
                generate_params,
                state_params,
                question
            )
        else:
            return BasicGenerator.generate(
                self,
                question,
                state_params,
                stopping_strings,
                eos_token,
                generate_params
            )