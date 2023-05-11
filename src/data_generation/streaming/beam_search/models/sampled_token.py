from dataclasses import dataclass
from typing import Union

import torch


@dataclass
class SampledToken:
    id: torch.Tensor
    logprob: Union[float, int]

    def __iter__(self):
        yield self.id
        yield self.logprob