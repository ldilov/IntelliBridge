from typing import Iterable, List

import ray

from kernel.modules.autocorrect.nltk_ngram.nltk_corpus import build_corpus
from kernel.modules.autocorrect.nltk_ngram.nltk_ngram_model import NLTKNgramModel
from kernel.modules.autocorrect.nltk_ngram.nltk_preprocess import normalize_corpus_text, tokenize_corpus_text, \
    clean_corpus_tokens


class NLTKTrainer:
    N_GRAMS = {
        2: "bigram",
        3: "trigram",
        4: "quadgram"
    }

    def __init__(self, n_gram_range: Iterable[int] = range(2, 5), language: str = "english"):
        if any(n > 4 for n in n_gram_range) or any(n < 2 for n in n_gram_range):
            raise ValueError("n_gram_range must be between 1 and 4")

        self.language: str = language
        self.n_gram_range: Iterable[int] = n_gram_range
        self.corpus: str = build_corpus()

        self._preprocess_corpus()

    def train(self) -> List[NLTKNgramModel]:
        ray.shutdown()
        ray.init(
            num_cpus=8,
            ignore_reinit_error=True,
            object_store_memory=16 * 1024 * 1024 * 1024,
        )

        results = [self.train_ngram.remote(n, self.corpus) for n in self.n_gram_range]
        n_gram_models: List[NLTKNgramModel] = ray.get(results)

        ray.shutdown()

        return n_gram_models

    def _preprocess_corpus(self):
        self.corpus = normalize_corpus_text(self.corpus)
        self.corpus = tokenize_corpus_text(self.corpus)
        self.corpus = clean_corpus_tokens(self.corpus)

    @staticmethod
    @ray.remote(num_cpus=2)
    def train_ngram(n, corpus_tokens):
        import psutil

        proc = psutil.Process()
        proc.cpu_affinity([n - 1, n + 1])

        model = NLTKNgramModel(n, corpus_tokens)
        model.train()
        return model
