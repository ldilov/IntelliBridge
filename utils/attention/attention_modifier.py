import transformers
from transformers import LlamaForCausalLM, GPTNeoXModel, GPTNeoXForCausalLM, GPTJForCausalLM, GPTJModel

from kernel.logger.logger import logger
from utils.attention.sdp_attn import sdp_attention_forward
from utils.attention.xformers_attn import xformers_forward


class AttentionModifier(object):
    def __init__(self, xformers=False, sdp_attention=False):
        xformers_forward.name = 'xformers'
        sdp_attention_forward.name = 'sdp'

        self.forward_fn = xformers_forward if xformers else sdp_attention_forward
        self.xformers = xformers
        self.sdp = sdp_attention

    def apply_llama(self):
        transformers.models.llama.modeling_llama.LlamaAttention.forward = self.forward_fn

    def apply_gptj(self):
        transformers.models.gptj.modeling_gptj.GPTJAttention.forward = self.forward_fn

    def apply_auto(self, model):
        if hasattr(model, 'module'):
            self.apply_auto(model.module)
            return

        if isinstance(model, LlamaForCausalLM):
            for layer in model.model.layers:
                layer.self_attn.forward = self.forward_fn.__get__(
                    layer.self_attn, layer.self_attn.__class__
                )
        elif isinstance(model, GPTNeoXForCausalLM):
            for layer in model.model.layers:
                layer.attention.forward = self.forward_fn.__get__(
                    layer.attention, layer.attention.__class__
                )
        elif isinstance(model, GPTJForCausalLM):
            for i in range(len(model.transformer.h)):
                model.transformer.h[i].attn.forward = self.forward_fn.__get__(
                    model.transformer.h[i].attn, model.transformer.h[i].attn.__class__
                )
        elif isinstance(model, GPTNeoXModel):
            for layer in model.layers:
                layer.attention.forward = self.forward_fn.__get__(
                    layer.attention, layer.attention.__class__
                )
        elif isinstance(model, GPTJModel):
            for i in range(len(model.transformer.h)):
                model.transformer.h[i].attn.forward = self.forward_fn.__get__(
                    model.transformer.h[i].attn, model.transformer.h[i].attn.__class__
                )
        else:
            raise NotImplementedError(f"Model {model.__class__.__name__} is not supported")

        attn_name = self.forward_fn.name if self.xformers or self.sdp else 'native'

        logger.success(f"Successfully applied attention modifier: {attn_name}")

