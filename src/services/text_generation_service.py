import os
import random

import numpy as np
import torch
import transformers
from pathlib import Path

from kernel.persistence.infra.models.abstract_model import AbstractModel
from src.data_generation.generators.basic_generator import BasicGenerator
from src.data_generation.generators.llamacpp_generator import LlamaCppGenerator
from src.data_processing.text_processor.galactica import Galactica
from src.data_processing.text_processor.gpt import GPT
from src.data_processing.text_processor.gpt4chan import Gpt4Chan
from utils.pytorch.gc import TorchGC


class TextGenerationService(object):
    def __init__(self, args, model: AbstractModel, soft_prompt=None, soft_prompt_tensor=None):
        self.model = model
        self.tokenizer = model.tokenizer
        self.soft_prompt = soft_prompt
        self.soft_prompt_tensor = soft_prompt_tensor
        self.is_llamacpp = "ggml" in Path(f'{args.model_dir}/{args.model}').name
        self.args = args
        self.gc = TorchGC(args)
        self.local_rank = self.args.local_rank if self.args.local_rank is not None else int(
            os.getenv("LOCAL_RANK", "0"))

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

    def generate_reply(self, input, state_params):
        if self.model is None:
            raise ModuleNotFoundError("No model is loaded! Select one in the Model tab.")
            return

        self.gc.clear_torch_cache()
        self.set_manual_seed(state_params['seed'])

        if self.is_llamacpp:
            return LlamaCppGenerator.generate(
                self.model,
                input,
                state_params,
            )
        else:
            return BasicGenerator.generate(
                self.model,
                input,
                state_params
            )
