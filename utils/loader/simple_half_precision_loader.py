import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, LlamaTokenizer

from kernel.logger.logger import logger
from utils.attention.attention_modifier import AttentionModifier
from utils.loader.abstract_loader import AbstractLoader


class SimpleFloatHalfPrecisionLoader(AbstractLoader):
    def __init__(self, name, dir, bf16=False, xformers=False, sdp_attention=False):
        super().__init__(name)
        self.dir = dir
        self.name = name
        self.bf16 = bf16
        self.xformers = xformers
        self.sdp_attention = sdp_attention

    def load(self):
        t0 = time.time()
        model = self.loader_class.from_pretrained(Path(f"{self.dir}/{self.name}"), low_cpu_mem_usage=True, torch_dtype=torch.bfloat16 if self.bf16 else torch.float16, trust_remote_code=self.trust_remote_code)
        if torch.has_mps:
            device = torch.device('mps')
            model = model.to(device)
        else:
            model = model.cuda()

        if any((self.xformers, self.sdp_attention)):
            AttentionModifier(self.xformers, self.sdp_attention).apply_auto(model)

        # Loading the tokenizer
        if any((k in self.name.lower() for k in ['gpt4chan', 'gpt-4chan'])) and Path(f"{self.dir}/gpt-j-6B/").exists():
            tokenizer = AutoTokenizer.from_pretrained(Path(f"{self.dir}/gpt-j-6B/"))
        elif type(model) is transformers.LlamaForCausalLM:
            tokenizer = LlamaTokenizer.from_pretrained(Path(f"{self.dir}/{self.name}/"), clean_up_tokenization_spaces=True)

            try:
                tokenizer.eos_token_id = 2
                tokenizer.bos_token_id = 1
                tokenizer.pad_token_id = 0
            except:
                pass
        else:
            tokenizer = AutoTokenizer.from_pretrained(Path(f"{self.dir}/{self.name}/"), trust_remote_code=self.trust_remote_code)

        logger.success(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
        return model, tokenizer