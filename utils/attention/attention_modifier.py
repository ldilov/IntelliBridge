import transformers

from kernel.logger.logger import logger
from utils.attention.sdp import sdp_attention_forward
from utils.attention.xformers import xformers_forward


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

        for i in range(len(model.transformer.h)):
            model.transformer.h[i].attn.forward = self.forward_fn.__get__(
                model.transformer.h[i].attn, model.transformer.h[i].attn.__class__
            )

        attn_name = self.forward_fn.name if self.xformers or self.sdp else 'native'

        logger.success(f"Successfully applied attention modifier: {attn_name}")

