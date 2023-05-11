from typing import List

import nltk
import ray
from nltk import FreqDist, MLEProbDist, ngrams


class NLTKNgramModel:
    def __init__(self, n, corpus_tokens):
        self.model: MLEProbDist = None
        self.n: int = n
        self.corpus_tokens: List[int] = corpus_tokens

    def get_probability(self, sequence: List[str]) -> float:
        sequence_ngrams = list(ngrams(sequence, self.n))

        probability = 1
        for ngram in sequence_ngrams:
            probability *= self.model.prob(ngram)

        return probability

    def train(self):
        ngram_fd = FreqDist(ngrams(self.corpus_tokens, self.n))
        self.model = MLEProbDist(ngram_fd)

    def select_best_sequence(self, candidates, models):
        best_sequence = None
        best_probability = -1

        for candidate in candidates:
            probability = 1
            for model in models:
                probability *= model.get_probability(candidate)

            if probability > best_probability:
                best_sequence = candidate
                best_probability = probability

        return best_sequence

    @staticmethod
    def save(models: List['NLTKNgramModel'], filename: str):
        with open(filename, 'wb') as f:
            import pickle
            pickle.dump(models, f)

    @staticmethod
    def load(filename: str) -> List['NLTKNgramModel']:
        with open(filename, 'rb') as f:
            import pickle
            models = pickle.load(f)

        return models