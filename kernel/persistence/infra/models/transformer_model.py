import json
import os
from pathlib import Path
from typing import Optional

import torch
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, LlamaModel, LlamaTokenizer, LlamaForCausalLM, \
    GenerationConfig, LogitsProcessorList, RepetitionPenaltyLogitsProcessor
from transformers.utils import PaddingStrategy

from kernel.persistence.infra.enums.model_type import ModelType
from kernel.persistence.infra.model_store import ModelStore
from kernel.persistence.infra.models.abstract_model import AbstractModel
from basaran.model import StreamModel

from kernel.persistence.storage.file_manager import FileManager
from utils.stopping_criteria.sentinel_token_stopping_criteria import SentinelTokenStoppingCriteria


class TransformerModel(AbstractModel):

    def __init__(self, model: Optional[PreTrainedModel], tokenizer: Optional[PreTrainedTokenizer], index_dict: dict[str,str]):
        super().__init__(model, tokenizer, index_dict)

    @classmethod
    def from_pretrained(cls, name, metadata_only=False, **kwargs):
        store = ModelStore()

        if metadata_only:
            model, path = store.load_with_metadata_only(name)
        else:
            model, path = store.load(name)

        model.path = path

        return model

    def generate(self, input_text):
        input_text = "BEGINNING OF CONVERSATION:\nUSER: " + input_text + "\nGPT:"

        from kernel.persistence.memory.global_modules_registry import registry as memory
        manager: FileManager = memory.get(FileManager)

        gen_config = json.loads(manager.read_file(self.config_path))
        gen_config = GenerationConfig(**gen_config)

        encoding = self.tokenizer.encode_plus(input_text, add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
        encoded = encoding['input_ids'].to('cuda')
        attn_mask = encoding['attention_mask'].to('cuda')
        self.model.cuda()

        stopping_criteria_list = transformers.StoppingCriteriaList()
        for st in ["USER: "]:
            sentinel_token_ids = [self.tokenizer.encode(st, add_special_tokens=False, return_tensors='pt').to('cuda:0')]
            stopping_criteria_list.append(SentinelTokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=len(encoded)))
            break

        logits_processor = LogitsProcessorList([
            RepetitionPenaltyLogitsProcessor(penalty=1.2)
        ])


        test = self.model.generate(input_ids=encoded, attention_mask=attn_mask, do_sample=True, stopping_criteria=stopping_criteria_list, logits_processor=logits_processor, generation_config=gen_config, eos_token_id=self.tokenizer.eos_token_id)
        a = self.tokenizer.decode(test[0], skip_special_tokens=True)

        for chunk in self.stream_model(input_text):
            print(chunk)
