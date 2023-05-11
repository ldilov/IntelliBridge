import sys
import typing
from typing import Protocol

from dependency_injector import containers, providers

from .coroutines import coroutines


class Coroutine(Protocol):
    """
    A protocol for coroutines that are to be injected into the container.
    Main entrypoint is the `init` method.
    """

    async def init(self) -> typing.Any:
        ...


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()


for coroutine in coroutines:  # type: Coroutine
    setattr(Container, coroutine.__name__.rsplit('.', 1)[-1], providers.Coroutine(coroutine.init))

__all__ = ['Container']
