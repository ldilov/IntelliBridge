import re
import time
from pathlib import Path

import psutil
import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from utils.attention.llama_attention import LlamaAttention
from utils.loader.abstract_loader import AbstractLoader


class NonquantizedLoader(AbstractLoader):
    def __init__(self, name=None, dir=None, xformers=False, sdp=False, gpu_memory: list = None, cpu_memory: list = None,
                 auto_devices=True, cpu=False, bf16=False, disk_cache_dir=None):
        super().__init__(name)

        self.name = name
        self.dir = dir
        self.cpu = cpu
        self.bf16 = bf16
        self.gpu_memory = gpu_memory
        self.cpu_memory = cpu_memory
        self.auto_devices = auto_devices
        self.xformers = xformers
        self.sdp = sdp

        total_memory = psutil.virtual_memory().total / 1024 / 1024 / 1024
        self.disk = total_memory < 32

    def load(self):
        t0 = time.time()

        params = {"low_cpu_mem_usage": True}
        if not any((self.cpu, torch.cuda.is_available(), torch.has_mps)):
            print(
                "Warning: torch.cuda.is_available() returned False.\nThis means that no GPU has been detected.\nFalling back to CPU mode.\n")
            self.cpu = True

        if self.cpu:
            params["torch_dtype"] = torch.float32
        else:
            params["device_map"] = 'auto'
            params["trust_remote_code"] = self.trust_remote_code
            if self.bf16:
                params["torch_dtype"] = torch.bfloat16
            else:
                params["torch_dtype"] = torch.float16

            if self.gpu_memory:
                memory_map = list(map(lambda x: x.strip(), self.gpu_memory))
                max_cpu_memory = self.cpu_memory.strip() if self.cpu_memory is not None else '64GiB'
                max_memory = {}
                for i in range(len(memory_map)):
                    max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else \
                    memory_map[i]
                max_memory['cpu'] = max_cpu_memory
                params['max_memory'] = max_memory
            elif self.auto_devices:
                total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
                suggestion = round((total_mem - 1000) / 1000) * 1000
                if total_mem - suggestion < 800:
                    suggestion -= 1000
                suggestion = int(round(suggestion / 1000))
                print(
                    f"\033[1;32;1mAuto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors.\nYou can manually set other values.\033[0;37;0m")

                max_memory = {0: f'{suggestion}GiB', 'cpu': f'{self.cpu_memory or 99}GiB'}
                params['max_memory'] = max_memory

            if self.disk:
                params["offload_folder"] = self.disk_cache_dir

        checkpoint = Path(f'{self.dir}/{self.name}')

        model = self.loader_class.from_pretrained(checkpoint, **params)

        if any((self.xformers, self.sdp)):
            LlamaAttention(self.xformers, self.sdp).hijack_llama_attention()

        # Loading the tokenizer
        if any((k in self.name.lower() for k in ['gpt4chan', 'gpt-4chan'])) and Path(f"{self.dir}/gpt-j-6B/").exists():
            tokenizer = AutoTokenizer.from_pretrained(Path(f"{self.dir}/gpt-j-6B/"))
        elif type(model) is transformers.LlamaForCausalLM:
            tokenizer = LlamaTokenizer.from_pretrained(Path(f"{self.dir}/{self.name}/"),
                                                       clean_up_tokenization_spaces=True)

            try:
                tokenizer.eos_token_id = 2
                tokenizer.bos_token_id = 1
                tokenizer.pad_token_id = 0
            except Exception as ex:
                print(f"Error: {ex}")
        else:
            tokenizer = AutoTokenizer.from_pretrained(Path(f"{self.dir}/{self.name}/"),
                                                      trust_remote_code=self.trust_remote_code)

        print(f"Loaded the model in {(time.time() - t0):.2f} seconds.")
        return model, tokenizer
