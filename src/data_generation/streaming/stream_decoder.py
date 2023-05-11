from typing import List

from src.data_generation.streaming.beam_search.models.beam_output_data import BeamOutputData
from src.data_generation.streaming.common.decorators.abstract_decorator import AbstractDecorator


class StreamDecoder:
    def __init__(self, decorators: List[AbstractDecorator]):
        self._decorators: List[AbstractDecorator] = decorators if decorators is not None else []
        self._sequence: str = ""

    def process(self, beam_output_data: BeamOutputData) -> BeamOutputData:
        for decorator in self._decorators:
            beam_output_data = decorator.process(beam_output_data, self._sequence)

        self._sequence += beam_output_data.decoded_output

        return beam_output_data

    def add_decorator(self, decorator: AbstractDecorator) -> None:
        decorator_type = type(decorator)
        existing_decorator_index: int = self.find_decorator_by(decorator_type)

        if existing_decorator_index is not None:
            self._decorators[existing_decorator_index] = decorator
        else:
            self._decorators.append(decorator)

    def extend_decorators(self, decorators: List[AbstractDecorator]) -> None:
        for decorator in decorators:
            self.add_decorator(decorator)

    def find_decorator_by(self, decorator_type):
        existing_decorator_index = None
        for i, existing_decorator in enumerate(self._decorators):
            if type(existing_decorator) == decorator_type:
                existing_decorator_index = i
                break
        return existing_decorator_index

