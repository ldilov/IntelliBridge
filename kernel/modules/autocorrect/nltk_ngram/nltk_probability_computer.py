import typing
from typing import List

import nltk
import ray

from kernel.modules.autocorrect.nltk_ngram.nltk_ngram_model import NLTKNgramModel


@ray.remote
class ProbabilityComputer:
    def __init__(self, models: List[NLTKNgramModel]):
        self.models = models

    def compute_probability(self, candidates: List[str]) -> typing.Dict[str, float]:
        candidate_probs = {}
        for model in self.models:
            for candidate in candidates:
                tokenized_candidate = nltk.word_tokenize(candidate)
                if candidate not in candidate_probs:
                    candidate_probs[candidate] = 1
                candidate_probs[candidate] *= model.get_probability(tokenized_candidate)
        return candidate_probs
