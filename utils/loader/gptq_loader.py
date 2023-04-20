import re
from pathlib import Path

import accelerate
import psutil
import torch
from transformers import AutoTokenizer

from kernel.logger.logger import logger
from utils.attention.attention_modifier import AttentionModifier
from utils.loader.abstract_loader import AbstractLoader
from utils.quantizer.engine.core import QuantCore
from utils.quantizer.engine.shared.lib import get_model_type


class GptQLoader(AbstractLoader):
    def __init__(self, args):
        super().__init__(args.model)
        self.args = args

        if not self.args.model_type:
            name = self.args.model.lower()
            model_type = get_model_type(name)
        else:
            model_type = get_model_type(self.args.model_type)

        self.quantizer = QuantCore(model_type)

    def load(self):
        # Find the quantized model weights file (.pt/.safetensors)
        path_to_model = Path(f'{self.args.model_dir}/{self.args.model}')
        pt_path = self._find_quantized_model_file(self.args.model)
        if not pt_path:
            logger.error("Could not find the quantized model in .pt or .safetensors format, exiting...")
            exit()
        else:
            with logger.contextualize(context=pt_path):
                logger.info(f"Found the following quantized model")

        extension = Path(path_to_model).suffix
        path_to_model = str(path_to_model).replace(extension, ".config.json")
        pt_path = str(pt_path)

        model = self.quantizer.load_quant(
            path_to_model,
            pt_path,
            self.args.wbits,
            self.args.groupsize,
            self.args.pre_layer
        )

        if self.args.use_accelerate:
            if self.args.gpu_memory or torch.cuda.device_count() > 1:
                from tempfile import gettempdir

                if self.args.gpu_memory:
                    memory_map = list(map(lambda x: x.strip(), self.args.gpu_memory))
                    max_physical_memory = str(int(psutil.virtual_memory().total / 1024 / 1024 / 1024))
                    max_cpu_memory = self.args.cpu_memory.strip() if self.args.cpu_memory is not None else max_physical_memory
                    max_memory = {}
                    for i in range(len(memory_map)):
                        max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
                    max_memory['cpu'] = max_cpu_memory
                else:
                    max_memory = accelerate.utils.get_balanced_memory(model)

                device_map = accelerate.infer_auto_device_map(model, max_memory=max_memory,  no_split_module_classes=["LlamaDecoderLayer"])
                logger.warning("Using the following device map for the quantized model:", device_map)
                model = accelerate.dispatch_model(model, device_map=device_map, offload_buffers=True, offload_dir=gettempdir())

        if any((self.args.xformers, self.args.sdp_attention)):
            attn_modifier = AttentionModifier(self.xformers, self.sdp_attention)
            attn_modifier.apply_auto(model)

        tokenizer = AutoTokenizer.from_pretrained("camelids/llama-65b-int4-gptq-groupsize128-safetensors")

        return model, tokenizer

    def _find_quantized_model_file(self, model_name):
        path_to_model = Path(f'{self.args.model_dir}/{model_name}')
        pt_path = None
        priority_name_list = [
            Path(f'{self.args.model_dir}/{model_name}{hyphen}{self.args.wbits}bit{group}{ext}')
            for group in ([f'-{self.args.groupsize}g', ''] if self.args.groupsize > 0 else [''])
            for ext in ['.safetensors', '.pt']
            for hyphen in ['-', f'/{model_name}-', '/']
        ]

        priority_name_list.append(path_to_model)

        for path in priority_name_list:
            if path.exists():
                pt_path = path
                break

        if not pt_path:
            found_pts = list(path_to_model.glob("*.pt"))
            found_safetensors = list(path_to_model.glob("*.safetensors"))
            pt_path = None

            if len(found_pts) > 0:
                if len(found_pts) > 1:
                    logger.warning('More than one .pt model has been found. The last one will be selected. It could be wrong.')
                pt_path = found_pts[-1]
            elif len(found_safetensors) > 0:
                if len(found_pts) > 1:
                    logger.warning('More than one .safetensors model has been found. The last one will be selected. It could be wrong.')
                pt_path = found_safetensors[-1]

        return pt_path
