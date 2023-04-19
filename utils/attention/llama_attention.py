import transformers

from utils.attention.sdp import sdp_attention_forward
from utils.attention.xformers import xformers_forward


class LlamaAttention(object):
    def __init__(self, xformers=False, sdp_attention=False):
        self.forward_fn = xformers_forward if xformers else sdp_attention_forward

    def hijack_llama_attention(self):
        transformers.models.llama.modeling_llama.LlamaAttention.forward = self.forward_fn
