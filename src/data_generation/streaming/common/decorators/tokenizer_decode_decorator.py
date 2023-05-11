from src.data_generation.streaming.beam_search.models.beam_output_data import BeamOutputData
from src.data_generation.streaming.common.decorators.abstract_decorator import AbstractDecorator
from src.data_generation.streaming.stream_tokenizer import StreamTokenizer


class TokenizerDecodeDecorator(AbstractDecorator):
    def __init__(self, tokenizer: StreamTokenizer):
        self.tokenizer = tokenizer

    def process(self, beam_output_data: BeamOutputData, sequence: str) -> BeamOutputData:
        decoded_output = self.tokenizer.decode(beam_output_data.output_token_id)
        beam_output_data.decoded_output = decoded_output
        return beam_output_data
