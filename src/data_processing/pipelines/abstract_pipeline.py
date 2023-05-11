from abc import ABC, abstractmethod

import torch


class AbstractPipeline(ABC):
    @abstractmethod
    def apply(self, input_ids: torch.Tensor, logits: torch.Tensor):
        raise NotImplementedError

    @property
    def processors(self):
        raise NotImplementedError
