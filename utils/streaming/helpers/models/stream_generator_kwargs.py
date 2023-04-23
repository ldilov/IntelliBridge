from dataclasses import dataclass

from utils.streaming.helpers.models.stream_generator_config import StreamGeneratorConfig


@dataclass
class StreamGeneratorKwargs:
    def __init__(self, config: StreamGeneratorConfig, input_ids, n):
        self.__dict = {
            "input_ids": input_ids,
            "do_sample": True,
            "num_return_sequences": n,
            "num_beams": 1,
            "use_cache": config.use_cache,
            "pad_token_id": config.pad_token_id,
            "bos_token_id": config.bos_token_id,
            "eos_token_id": config.eos_token_id,
            "decoder_start_token_id": config.decoder_start_token_id
        }

    def __getitem__(self, key):
        return self.__dict.get(key, None)

    def __iter__(self):
        return iter(self.__dict.items())

    def items(self):
        return self.__dict.items()
