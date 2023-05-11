from abc import ABC

from src.data_generation.streaming.beam_search.models.beam_output_data import BeamOutputData


class AbstractDecorator(ABC):
    def process(self, beam_output_data: BeamOutputData, sequence: str) -> BeamOutputData:
        raise NotImplementedError("process() method must be implemented in the subclass")
