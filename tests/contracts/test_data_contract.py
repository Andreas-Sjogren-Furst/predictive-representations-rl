from collections.abc import Iterator

from aeris.contracts import BatchProvider, TransitionSource
from aeris.domain import Transition


class MemoryTransitions:
    def __init__(self, experiences: list[Transition]) -> None:
        self.data = experiences

    def iter_transitions(self) -> Iterator[Transition]:
        yield from self.data


class PairedBatchProvider:
    def __init__(self, source: TransitionSource) -> None:
        self.data = list(source.iter_transitions())

    def next_batch(self) -> tuple[Transition, Transition]:
        return self.data[0], self.data[-1]


class SequenceBatchProvider:
    """Demonstrate a different batch shape without changing TransitionSource."""

    def __init__(self, source: TransitionSource) -> None:
        self.data = list(source.iter_transitions())

    def next_batch(self) -> list[Transition]:
        return self.data[:]


def test_same_source_supports_paired_and_sequence_sampling():
    source = MemoryTransitions([Transition(0, 0, 1), Transition(1, 1, 2), Transition(2, 0, 3)])
    assert isinstance(source, TransitionSource)
    pair = PairedBatchProvider(source)
    seq = SequenceBatchProvider(source)
    assert isinstance(pair, BatchProvider)
    assert isinstance(seq, BatchProvider)
    assert len(pair.next_batch()) == 2
    assert len(seq.next_batch()) == 3
    assert source.data[0].reward is None
