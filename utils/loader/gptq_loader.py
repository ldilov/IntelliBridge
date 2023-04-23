import os
from pathlib import Path
from transformers import AutoTokenizer

from kernel.logger.logger import logger
from kernel.persistence.infra.models.gptq_model import GptqModel
from kernel.persistence.memory.plugins import Plugins
from utils.attention.attention_modifier import AttentionModifier
from utils.loader.abstract_loader import AbstractLoader
from kernel.persistence.memory.global_registry import registry as memory
from kernel.persistence.memory.global_modules_registry import registry as modules
from utils.quantizer.adapter import AdapterGPTQForCausalLm
from utils.third_party.AutoGPTQ.auto_gptq import AutoGPTQForCausalLM


class GptQLoader(AbstractLoader):
    def __init__(self):
        args = memory.get('args')
        super().__init__(args.model)
        self.args = args
        self.plugins: Plugins = modules.get(Plugins)
        self.autogptq = self.plugins.get_plugin("AutoGPTQ")

    def load(self) -> GptqModel:
        path_to_model = Path(self.args.model)
        gptq_model: GptqModel = GptqModel.from_pretrained(path_to_model)

        if any((self.args.xformers, self.args.sdp_attention)):
            attn_modifier = AttentionModifier(self.args.xformers, self.args.sdp_attention)
            attn_modifier.apply_auto(gptq_model.model)

        return gptq_model
