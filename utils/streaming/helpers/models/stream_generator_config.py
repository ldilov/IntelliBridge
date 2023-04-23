from dataclasses import dataclass
from typing import Optional, List


@dataclass
class StreamGeneratorConfig:
    min_new_tokens: int = 0
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    decoder_start_token_id: Optional[int] = None
    no_repeat_ngram_size: int = 0
    repetition_penalty: float = 1.0
    min_top_p: float = 0.1
    max_top_p: float = 0.9
    top_p_transition_length: int = 50
    top_p_scale: float = 1.0
    min_temperature: float = 1.0
    max_temperature: float = 1.5
    temperature_transition_length: int = 50
    topic_bias_keywords: Optional[List[int]] = None
    topic_bias_weight: float = 0.0
    creative_boost_weight: float = 0.0
    rare_token_ids: Optional[List[int]] = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self