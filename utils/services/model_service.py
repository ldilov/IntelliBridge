import json
import re
import zipfile
from pathlib import Path

import numpy as np
import torch

from kernel.logger.logger import logger
from kernel.persistence.infra.models.abstract_model import AbstractModel
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
        if self.args.deepspeed:
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
            gptq_loader = GptQLoader()

            if self.args.monkey_patch:
                from utils.loader.loader_4bit_helper import load_model_llama_4bit
                loader = type('', (object,), {"load": lambda: load_model_llama_4bit(self.args, gptq_loader)})()
            else:
                loader = gptq_loader
        else:
            from utils.loader.full_precision_loader import FullPrecisionLoader
            loader = FullPrecisionLoader(
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

    def load_model(self) -> AbstractModel:
        model: AbstractModel = self.loader.load()

        return model

    def unload_model(self):
        del self.model_ref
        del self.tokenizer_ref

        self.model_ref = self.tokenizer_ref = None
        self.gc.clear_torch_cache()

    def reload_model(self) -> AbstractModel:
        self.unload_model()
        return self.load_model()