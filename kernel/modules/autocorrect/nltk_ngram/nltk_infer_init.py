import asyncio
import typing

import ray
from typing import List

from kernel.modules.autocorrect.nltk_ngram.nltk_ngram_model import NLTKNgramModel
from kernel.modules.autocorrect.nltk_ngram.nltk_probability_computer import ProbabilityComputer


def init_ray():
    ray.shutdown()
    ray.init(
        local_mode=False,
        num_cpus=4,
        ignore_reinit_error=True,
        object_store_memory=16 * 1024 * 1024 * 1024,
    )


def shutdown_ray():
    ray.shutdown()


async def create_actor(models: List[NLTKNgramModel]):
    return ProbabilityComputer.remote(models)


async def create_actors_async(models: List[NLTKNgramModel]):
    tasks = [create_actor(models) for _ in range(4)]
    return await asyncio.gather(*tasks)


async def init() -> List[ProbabilityComputer]:
    models: List[NLTKNgramModel] = NLTKNgramModel.load("nltk_ngram_model_english.pkl")
    init_ray()
    loop = asyncio.get_event_loop()
    probability_computers = loop.run_until_complete(create_actors_async(models))
    return probability_computers

__all__: List[List[ProbabilityComputer], typing.Callable[[], None]] = init()
