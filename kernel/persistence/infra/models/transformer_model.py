import json
from typing import Optional

import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig

from kernel.persistence.infra.model_store import ModelStore
from kernel.persistence.infra.models.abstract_model import AbstractModel

from kernel.persistence.storage.file_manager import FileManager
from src.data_processing.pipelines.abstract_pipeline import AbstractPipeline
from src.data_processing.pipelines.output_pipeline import OutputLogitsPipeline
from src.data_generation.stopping_criteria.sentinel_token_stopping_criteria import SentinelTokenStoppingCriteria
from src.data_generation.streaming.stream_generator import StreamGenerator
from src.data_generation.streaming.common.samplers.hybrid import HybridTokenSampler


class TransformerModel(AbstractModel):

    def __init__(self, model: Optional[PreTrainedModel], tokenizer: Optional[PreTrainedTokenizer], index_dict: dict[str,str]):
        super().__init__(model, tokenizer, index_dict)

        self._sampler = HybridTokenSampler(self.tokenizer, self._generation_config)
        self._logits_pipe: AbstractPipeline = OutputLogitsPipeline(self._generation_config)
        args = (self._model, self._tokenizer, self._sampler, self._generation_config, self._logits_pipe)
        self._stream_generator: StreamGenerator = StreamGenerator(*args)

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

        for chunk in self._stream_generator(input_text):
            print(chunk, end='')

