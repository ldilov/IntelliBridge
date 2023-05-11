from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class StreamGeneratorConfig(Mapping, ABC):
    min_new_tokens: int = 15
    max_new_tokens: int = 50
    do_sample: bool = True
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    unk_token_id: Optional[int] = None
    decoder_start_token_id: Optional[int] = None
    no_repeat_ngram_size: int = 0
    repetition_penalty: float = 1.0
    min_top_p: float = 0.3
    max_top_p: float = 0.9
    top_p_transition_length: int = 4
    top_p_scale: float = 1.0
    top_k: int = 0
    top_p: int = 95
    temperature: float = 1.0
    min_temperature: float = 1.0
    max_temperature: float = 1.5
    temperature_transition_length: int = 50
    topic_bias_keywords: Optional[List[int]] = None
    topic_bias_weight: float = 0.0
    length_penalty: float = 1.0
    creative_boost_weight: float = 0.0
    rare_token_ids: Optional[List[int]] = None
    k_beam: int = 4
    beam_size: int = 5
    max_length: int = 50
    chunk_size: int = 5
    prob_threshold: float = 0.1

    def __init__(self, **kwargs):
        super().__init__()
        self.update(**kwargs)

    def __getitem__(self, key):
        return self.__dict.get(key, None)

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)

    def __len__(self):
        return len(self.__dict)

    def __iter__(self):
        return iter(self.__dict.items())

    def as_dict(self):
        return self.__dict__

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
