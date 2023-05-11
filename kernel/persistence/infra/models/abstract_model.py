import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Optional

from transformers import PreTrainedModel, PreTrainedTokenizer

from kernel.persistence.infra.enums.model_type import ModelType
from kernel.persistence.infra.model_store import ModelStore
from src.data_generation.streaming.stream_generator_config import StreamGeneratorConfig
from src.data_generation.streaming.stream_generator import StreamGenerator
from src.data_generation.streaming.common.samplers.hybrid import HybridTokenSampler


class AbstractModel(ABC):
    def __init__(self, model: Optional[PreTrainedModel], tokenizer: Optional[PreTrainedTokenizer], index_dict: dict[str,str]):
        self._model: Optional[PreTrainedModel] = model
        self._tokenizer: Optional[PreTrainedTokenizer] = tokenizer
        self._type: ModelType = ModelType.CUSTOM
        self._stream_generator: Optional[StreamModel] = None
        self._config_path: Path = Path(index_dict['config'])
        self._tokenizer_config_path: Path = Path(index_dict['tokenizer'])
        self._name: str = str(index_dict['name'])
        self._extension: str = str(index_dict['extension'])
        self._full_path: Path = Path(os.getcwd()) / 'resources' / 'models' / self._name

        if index_dict['model_type'].lower() == "gpt":
            self._type: ModelType = ModelType.GPT
        elif index_dict['model_type'].lower() == "llama":
            self._type: ModelType = ModelType.LLAMA

        self.__store = ModelStore()
        generator_config_dict = self.__store.load_generation_config(self._full_path)
        self._generation_config: StreamGeneratorConfig = StreamGeneratorConfig(**generator_config_dict)
        if self._model is not None and self._tokenizer is not None:
            sampler = HybridTokenSampler(self._generation_config)
            self._stream_generator: StreamGenerator = StreamGenerator(self._model, self._tokenizer, sampler, self._generation_config)

    @abstractmethod
    def generate(self, input_text: str) -> Generator[int, None, None]:
        pass

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    @model.setter
    def model(self, model: PreTrainedModel):
        self._model = model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizer):
        self._tokenizer = tokenizer

    @property
    def type(self) -> ModelType:
        return self._type

    @property
    def config_path(self) -> Path:
        return self._config_path

    @property
    def tokenizer_config_path(self) -> Path:
        return self._tokenizer_config_path

    @property
    def name(self) -> str:
        return self._name

    @property
    def extension(self) -> str:
        return self._extension

    @property
    def full_path(self) -> Path:
        return self._full_path

    @property
    def generation_config(self) -> GenerationConfig:
        return self._generation_config

    @property
    def type(self, value):
        self._type = value

    @type.setter
    def type(self, value):
        self._type = value

    @generation_config.setter
    def generation_config(self, generation_config: GenerationConfig):
        self._generation_config = generation_config


