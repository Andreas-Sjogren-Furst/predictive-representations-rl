from collections.abc import Iterator
from typing import Protocol, TypeVar, runtime_checkable

from aeris.domain.experience import Transition

BatchT_co = TypeVar("BatchT_co", covariant=True)


@runtime_checkable
class TransitionSource(Protocol):
    def iter_transitions(self) -> Iterator[Transition]:
        ...

@runtime_checkable
class BatchProvider(Protocol[BatchT_co]):
    def next_batch(self) -> BatchT_co:
        ...
