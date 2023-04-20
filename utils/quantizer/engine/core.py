import os
from pathlib import Path

import torch
import transformers
from accelerate import init_empty_weights
from torch import nn, device
from transformers import LlamaConfig, LlamaForCausalLM

from kernel.logger.logger import logger
from utils.quantizer.engine.enums.model_type import ModelType
from utils.quantizer.engine.llama.layers.q_linear import QLinear as QLlamaLinear
from utils.quantizer.engine.gpt.layers.q_linear import QLinear as QGptLinear
from utils.quantizer.engine.llama.modules.m_offload import MOffload as LlamaMOffload
from utils.quantizer.engine.gpt.modules.m_offload import MOffload as GptjMOffload


class QuantCore(object):
    def __init__(self, model_type: ModelType, offload_device: device = torch.device('cpu')):
        if not issubclass(type(model_type), ModelType):
            raise ValueError(f'model_type must be of type {ModelType.__name__}')

        self.is_cuda = False
        self.model_type = model_type

        torch.nn.init.kaiming_uniform_ = self._skip
        torch.nn.init.uniform_ = self._skip
        torch.nn.init.normal_ = self._skip

        torch.set_default_dtype(torch.half)
        transformers.modeling_utils._init_weights = False

        if offload_device.type == 'cpu':
            if model_type == ModelType.LLAMA:
                transformers.models.llama.modeling_llama.LlamaModel = LlamaMOffload
            elif model_type == ModelType.GPTJ:
                transformers.models.gptj.modeling_gptj.GPTJModel = GptjMOffload
            else:
                raise ValueError('Unknown model type')

        self.device = torch.device("cuda") if self.is_cuda else torch.device("cpu")

    def load_quant(self, model, checkpoint, wbits, groupsize, pre_layer=None):
        torch.set_default_dtype(torch.half)

        with init_empty_weights():
            config = LlamaConfig.from_json_file(model)
            model = LlamaForCausalLM(config)
            torch.set_default_dtype(torch.float)
            model = model.eval()

        layers = self.find_layers(model)
        for name in ['lm_head']:
            if name in layers:
                del layers[name]

        self.make_quant(model, layers, wbits, groupsize)

        logger.info("Loading model...")
        if checkpoint.endswith('.safetensors'):
            from safetensors.torch import load_file as safe_load
            model.load_state_dict(safe_load(checkpoint, f'{self.device}'), strict=False)
        else:
            model.load_state_dict(torch.load(checkpoint), strict=False)
        logger.success("Model loaded successfully!")

        model.seqlen = 2048

        if pre_layer is not None and self.model_type == ModelType.LLAMA:
            gpu_device = torch.device("cuda:0")

            for i in range(pre_layer):
                model.model.layers[i].to(gpu_device)

            model.model.embed_tokens.to(gpu_device)
            model.model.norm.to(gpu_device)
            model.lm_head.to(gpu_device)
            model.model.preload = pre_layer

        return model

    def get_llama(self, model):
        model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
        model.seqlen = 2048

        return model

    def find_layers(self, module, layers=[nn.Conv2d, nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}

        res = {}
        for name1, child in module.named_children():
            res.update(
                self.find_layers(
                    child, layers=layers, name=name + '.' + name1 if name != '' else name1
                )
            )
        return res

    def make_quant(self, module, names, bits, groupsize, name=''):
        if isinstance(module, QLlamaLinear) or isinstance(module, QGptLinear):
            return

        for attr in dir(module):
            tmp = getattr(module, attr)
            name1 = name + '.' + attr if name != '' else attr

            if name1 in names:
                if self.model_type == ModelType.LLAMA:
                    q_linear_layer = QLlamaLinear(
                        bits, groupsize, tmp.in_features,
                        tmp.out_features, tmp.bias is not None, is_cuda=self.is_cuda
                    )
                elif self.model_type == ModelType.GPTJ:
                    q_linear_layer = QGptLinear(
                        bits, groupsize, tmp.in_features, tmp.out_features
                    )
                else:
                    raise NotImplementedError(f'Quantization for {self.model_type} is not implemented yet.')

                delattr(module, attr)
                setattr(module, attr, q_linear_layer)
        for name1, child in module.named_children():
            self.make_quant(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)

    def _skip(*args, **kwargs):
        pass

    def initialize_cuda_extension(self):
        global_initialize_cuda_extension()


def global_initialize_cuda_extension():
    try:
        os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
        global quant_cuda
        import quant_cuda
    except:
        logger.error('CUDA extension not installed.')
