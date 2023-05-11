import typing
import asyncio
from typing import List

import ray
from spellchecker import SpellChecker
from kernel.modules.autocorrect.nltk_ngram.nltk_ngram_model import NLTKNgramModel
from kernel.modules.autocorrect.nltk_ngram.nltk_probability_computer import ProbabilityComputer


class NLTKInfer:
    @staticmethod
    def generate_candidates(sentence: str, max_edit_distance: int = 2) -> List[str]:
        spell = SpellChecker()
        words = sentence.split()

        def generate_word_candidates(word: str, distance: int) -> List[str]:
            if distance <= 0:
                return [word]

            first_order_candidates = spell.candidates(word)
            higher_order_candidates = set()
            for candidate in first_order_candidates:
                higher_order_candidates.update(generate_word_candidates(candidate, distance - 1))
            return list(higher_order_candidates)

        def generate_sentence_candidates(words: List[str], idx: int) -> List[str]:
            if idx >= len(words):
                return [' '.join(words)]

            current_word_candidates = generate_word_candidates(words[idx], max_edit_distance)
            sentences = []

            for candidate_word in current_word_candidates:
                new_words = words[:idx] + [candidate_word] + words[idx + 1:]
                sentences.extend(generate_sentence_candidates(new_words, idx + 1))

            return sentences

        candidates = generate_sentence_candidates(words, 0)
        return candidates

    @staticmethod
    def select_best_sequence(candidates: List[List[str]], models: List[NLTKNgramModel]):
        ray.shutdown()
        ray.init(
            local_mode=False,
            num_cpus=4,
            ignore_reinit_error=True,
            object_store_memory=16 * 1024 * 1024 * 1024,
        )

        async def create_actor(models):
            return ProbabilityComputer.remote(models)

        async def main():
            nonlocal best_sequence, best_probability

            num_cores = 4
            batch_size = len(candidates) // num_cores

            # Create ProbabilityComputer actors
            tasks = [create_actor(models) for _ in range(num_cores)]
            probability_computers = await asyncio.gather(*tasks)

            # Distribute the workload among the actors
            futures = [pc.compute_probability.remote(candidates[i:i + batch_size]) for i, pc in
                       zip(range(0, len(candidates), batch_size), probability_computers)]

            while len(futures) > 0:
                ready_futures, futures = ray.wait(futures)
                probabilities: typing.Dict[str, float] = ray.get(ready_futures)[0]
                best_batch_seq, best_batch_prob = max(probabilities.items(), key=lambda x: x[1])

                if best_batch_prob > best_probability:
                    best_probability = best_batch_prob
                    best_sequence = best_batch_seq

            ray.shutdown()

        best_sequence = None
        best_probability = -1
        asyncio.run(main())

        return best_sequence
