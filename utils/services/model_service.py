import json
import re
import zipfile
from pathlib import Path

import numpy as np
import torch

from utils.pytorch.gc import TorchGC


class ModelService(object):
    def __init__(self, args):
        self.args = args
        self.model_name = args.model
        self.gc = TorchGC(args)

        self.model_ref = None
        self.tokenizer_ref = None

        self.soft_prompt = False
        self.soft_prompt_tensor = None
        self.is_llamacpp = "ggml" in Path(f'{self.args.model_dir}/{self.model_name}').name

        loader = None
        if not any([self.args.cpu, self.args.auto_devices, self.args.gpu_memory is not None,
                    self.args.cpu_memory is not None, self.args.deepspeed, self.is_llamacpp]):
            from utils.loader.simple_half_precision_loader import SimpleFloatHalfPrecisionLoader
            loader = SimpleFloatHalfPrecisionLoader(
                self.model_name,
                self.args.model_dir,
                self.args.bf16,
                self.args.xformers,
                self.args.sdp_attention
            )
        elif self.args.deepspeed:
            from utils.loader.deepspeed_loader import DeepSpeedLoader
            loader = DeepSpeedLoader(
                self.model_name,
                self.args.model_dir,
                self.args.nvme_offload_dir,
                self.args.bf16,
                self.args.xformers,
                self.args.sdp_attention,
            )
        elif self.is_llamacpp:
            from utils.loader.llamacpp_loader import LlamaCppLoader
            loader = LlamaCppLoader(
                self.model_name,
                self.args.model_dir
            )

        elif self.args.wbits > 0:
            from utils.loader.gptq_loader import GptQLoader
            gptq_loader = GptQLoader(self.args)

            if self.args.monkey_patch:
                from utils.loader.loader_4bit_helper import load_model_llama_4bit
                loader = type('', (object,), {"load": lambda: load_model_llama_4bit(self.args, gptq_loader)})()
            else:
                loader = gptq_loader
        else:
            from utils.loader.nonquantized_loader import NonquantizedLoader
            loader = NonquantizedLoader(
                self.model_name,
                self.args.model_dir,
                self.args.xformers,
                self.args.sdp_attention,
                self.args.gpu_memory,
                self.args.cpu_memory,
                self.args.auto_devices,
                self.args.cpu,
                self.args.bf16,
                self.args.disk_cache_dir
            )

        self.loader = loader

    def load_model(self):
        self.model_ref, self.tokenizer_ref = self.loader.load()

        return self.model_ref, self.tokenizer_ref

    def unload_model(self):
        del self.model_ref
        del self.tokenizer_ref

        self.model_ref = self.tokenizer_ref = None
        self.gc.clear_torch_cache()

    def reload_model(self):
        self.unload_model()
        self.model_ref, self.tokenizer_ref = self.load_model()

    def load_soft_prompt(self, name):
        if name == 'None':
            self.soft_prompt = False
            self.soft_prompt_tensor = None
        else:
            with zipfile.ZipFile(Path(f'softprompts/{name}.zip')) as zf:
                zf.extract('tensor.npy')
                zf.extract('meta.json')
                j = json.loads(open('meta.json', 'r').read())
                print(f"\nLoading the softprompt \"{name}\".")
                for field in j:
                    if field != 'name':
                        if type(j[field]) is list:
                            print(f"{field}: {', '.join(j[field])}")
                        else:
                            print(f"{field}: {j[field]}")
                print()
                tensor = np.load('tensor.npy')
                Path('tensor.npy').unlink()
                Path('meta.json').unlink()
            tensor = torch.Tensor(tensor).to(device=self.model_ref.device, dtype=self.model_ref.dtype)
            tensor = torch.reshape(tensor, (1, tensor.shape[0], tensor.shape[1]))

            self.soft_prompt = True
            self.soft_prompt_tensor = tensor

        return name

    def get_soft_prompt(self):
        return self.soft_prompt, self.soft_prompt_tensor
