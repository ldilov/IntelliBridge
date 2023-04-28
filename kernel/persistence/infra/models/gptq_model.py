import re
from pathlib import Path

from basaran.model import StreamModel
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from kernel.persistence.infra.model_store import ModelStore
from kernel.persistence.memory.global_modules_registry import registry as memory
from kernel.persistence.infra.models.transformer_model import TransformerModel
from kernel.persistence.storage.file_manager import FileManager
from utils.quantizer.adapter import AdapterGPTQForCausalLm
from utils.streaming.stream_generator import StreamGenerator
from utils.third_party.AutoGPTQ.auto_gptq import BaseQuantizeConfig
from utils.third_party.AutoGPTQ.auto_gptq.modeling.auto import AutoGPTQForCausalLM


class GptqModel(TransformerModel):
    def __init__(self, model, tokenizer, index_dict):
        super().__init__(model, tokenizer, index_dict)
        self.type = index_dict.get('type', None)

    def generate(self, input_text):
        return super().generate(input_text)

    def build_quantize_config(self):
        quantize_config = BaseQuantizeConfig(
            bits=self.get_wbits(),
            group_size=self.get_group_size(),
            desc_act=True
        )

        file_manager: FileManager = memory.get(FileManager)

        if not file_manager.exists(self.full_path / 'quantize_config.json'):
            quantize_config.save_pretrained(self.full_path)

        return quantize_config

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        gptq_model: GptqModel = super().from_pretrained(name, metadata_only=True)

        if kwargs.get('model', None) and kwargs.get('tokenizer', None):
            gptq_model.model = kwargs['model']
            gptq_model.tokenizer = kwargs['tokenizer']
        else:
            AdapterGPTQForCausalLm.patch()
            gptq_model.build_quantize_config()
            gptq_causallm_model: AdapterGPTQForCausalLm = AutoGPTQForCausalLM.from_quantized(str(gptq_model.full_path))
            gptq_model.model: PreTrainedModel = gptq_causallm_model.model
            gptq_model.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                gptq_model.full_path,
                use_fast=False,
                trust_remote_code=True
            )

        gptq_model._stream_generator = StreamGenerator(gptq_model.model, gptq_model.tokenizer, gptq_model._generation_config)

        return gptq_model

    def is_safetensors(self):
        return self.full_path.lower().endswith('safetensors')

    def get_wbits(self):
        found_wbits = re.search(r'(int(?P<wbits1>[48])|(?P<wbits2>[48])-?bit)', self.name, re.I)

        if found_wbits:
            return int(found_wbits.group('wbits1') or found_wbits.group('wbits2'))

        return None

    def get_group_size(self):
        match = re.search(r'((?P<group_size>\d{1,3})g|groupsize(?P<group_size2>\d{1,3}))', self.name, re.I)
        group_size = None

        if match is not None:
            group_size = int(match.group('group_size')) if match.groupdict()['group_size'] else int(
                match.group('group_size2'))

        return group_size