import os
from typing import Optional

import torch
import transformers

from kernel.logger.logger import logger
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel
from kernel.persistence.memory.global_registry import registry as memory
from kernel.performance.intelli_flow_accelerator import IntelliFlowAccelerator
from external.plugins.AutoGPTQ.auto_gptq import BaseQuantizeConfig
from external.plugins.AutoGPTQ.auto_gptq.modeling import BaseGPTQForCausalLM, GPTNeoXGPTQForCausalLM, \
    GPTJGPTQForCausalLM, LlamaGPTQForCausalLM, OPTGPTQForCausalLM
from external.plugins.AutoGPTQ.auto_gptq.modeling.auto import GPTQ_CAUSAL_LM_MODEL_MAP
from external.plugins.AutoGPTQ.auto_gptq.modeling._const import SUPPORTED_MODELS
from external.plugins.AutoGPTQ.auto_gptq.modeling._utils import find_layers, make_quant
from os.path import join
from pathlib import Path


class AdapterGPTQForCausalLm(BaseGPTQForCausalLM):

    def __init__(self, model: PreTrainedModel, quantized: bool, quantize_config: BaseQuantizeConfig):
        super().__init__(model, quantized, quantize_config)

    @classmethod
    def from_quantized(
            cls,
            save_dir: str,
            device: str = "cpu",
            use_safetensors: bool = False,
            use_triton: bool = False,
            max_memory: Optional[dict] = None,
            device_map: Optional[str] = None
    ):
        """load quantized model from local disk"""
        config = AutoConfig.from_pretrained(save_dir)
        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        quantize_config = BaseQuantizeConfig.from_pretrained(save_dir)
        checkpoint_file = cls.__find_quantized_model_file(Path(save_dir).name)
        model_save_name = join(save_dir, checkpoint_file)

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        transformers.modeling_utils._init_weights = False
        torch.set_default_dtype(torch.half)
        model = AutoModelForCausalLM.from_config(config)
        torch.set_default_dtype(torch.float)
        model = model.eval()
        layers = find_layers(model)
        for name in ['lm_head']:
            if name in layers:
                del layers[name]
        make_quant(model, layers, quantize_config.bits, quantize_config.group_size)

        accelerator = IntelliFlowAccelerator()
        if model_save_name.endswith('.safetensors'):
            if memory.get("args").use_accelerate:
                logger.warning("Loading model with accelerate")
                from safetensors.torch import load_file as safe_load
                model = accelerator.accelerate(model, model_save_name)
            else:
                from safetensors.torch import load_file as safe_load
                model.load_state_dict(safe_load(checkpoint_file), strict=False)

        else:
            model.load_state_dict(torch.load(checkpoint_file), strict=False)

        model.seqlen = model.config.max_position_embeddings

        model.eval()
        model.to(device)

        return cls(model, True, quantize_config)

    @classmethod
    def patch(cls):
        """patch AutoGPTQForCausalLM to support quantized model"""
        # global_initialize_cuda_extension()

        NewGPTNeoXGPTQForCausalLM: GPTNeoXGPTQForCausalLM = type("GPTNeoXGPTQForCausalLM",
                                                                 (AdapterGPTQForCausalLm, GPTNeoXGPTQForCausalLM),
                                                                 dict(GPTNeoXGPTQForCausalLM.__dict__))

        NewGPTJGPTQForCausalLM: GPTJGPTQForCausalLM = type("GPTJGPTQForCausalLM",
                                                           (AdapterGPTQForCausalLm, GPTJGPTQForCausalLM),
                                                           dict(GPTJGPTQForCausalLM.__dict__))

        NewLlamaGPTQForCausalLM: LlamaGPTQForCausalLM = type("LlamaGPTQForCausalLM",
                                                             (AdapterGPTQForCausalLm, LlamaGPTQForCausalLM),
                                                             dict(LlamaGPTQForCausalLM.__dict__))

        NewOptGPTQForCausalLM: OPTGPTQForCausalLM = type("OPTGPTQForCausalLM",
                                                         (AdapterGPTQForCausalLm, OPTGPTQForCausalLM),
                                                         dict(OPTGPTQForCausalLM.__dict__))

        GPTQ_CAUSAL_LM_MODEL_MAP["gpt_neox"] = NewGPTNeoXGPTQForCausalLM
        GPTQ_CAUSAL_LM_MODEL_MAP["gptj"] = NewGPTJGPTQForCausalLM
        GPTQ_CAUSAL_LM_MODEL_MAP["llama"] = NewLlamaGPTQForCausalLM
        GPTQ_CAUSAL_LM_MODEL_MAP["opt"] = NewOptGPTQForCausalLM

    @classmethod
    def __find_quantized_model_file(cls, model_name):
        path_to_model = Path(os.getcwd()) / 'resources' / 'models' / model_name

        found_pts = list(path_to_model.glob("*.pt"))
        found_safetensors = list(path_to_model.glob("*.safetensors"))

        if len(found_pts) > 0:
            if len(found_pts) > 1:
                logger.warning(
                    'More than one .pt model has been found. The last one will be selected. It could be wrong.')
            pt_path = found_pts[-1]
        elif len(found_safetensors) > 0:
            if len(found_pts) > 1:
                logger.warning(
                    'More than one .safetensors model has been found. The last one will be selected. It could be wrong.')
            pt_path = found_safetensors[-1]

        return pt_path


def global_initialize_cuda_extension():
    try:
        os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
        global quant_cuda
        import quant_cuda
    except:
        logger.error('CUDA extension not installed.')


NewGPTNeoXGPTQForCausalLM = type("GPTNeoXGPTQForCausalLM", (AdapterGPTQForCausalLm, GPTNeoXGPTQForCausalLM),
                                 dict(GPTNeoXGPTQForCausalLM.__dict__))

NewGPTJGPTQForCausalLM = type("GPTJGPTQForCausalLM", (AdapterGPTQForCausalLm, GPTJGPTQForCausalLM),
                              dict(GPTJGPTQForCausalLM.__dict__))

NewLlamaGPTQForCausalLM = type("LlamaGPTQForCausalLM", (AdapterGPTQForCausalLm, LlamaGPTQForCausalLM),
                               dict(LlamaGPTQForCausalLM.__dict__))

GPTQ_CAUSAL_LM_MODEL_MAP["gpt_neox"] = NewGPTNeoXGPTQForCausalLM
GPTQ_CAUSAL_LM_MODEL_MAP["gptj"] = NewGPTJGPTQForCausalLM
GPTQ_CAUSAL_LM_MODEL_MAP["llama"] = NewLlamaGPTQForCausalLM

__all__ = [NewGPTNeoXGPTQForCausalLM, NewGPTJGPTQForCausalLM, NewLlamaGPTQForCausalLM, AdapterGPTQForCausalLm]
