# Code originally belongs to the LLaMA GPTQ implementation at
#     https://github.com/qwopqwop200/GPTQ-for-LLaMa
# I've adapted it to work with GPT-J.
import transformers

from .modelutils import *
from .quant import *

DEV = torch.device('cuda:0')


def get_gptj(model):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import GPTJForCausalLM
    model = GPTJForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    model.to(DEV)
    return model


def load_quant(model, checkpoint, wbits):
    from transformers import GPTJConfig, GPTJForCausalLM
    config = GPTJConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = GPTJForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits)

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)
    model.seqlen = 2048
    print('Done.')

    return model


def get_tokenizer(model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer
