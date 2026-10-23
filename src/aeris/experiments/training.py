"""AERIS training loop, intentionally agnostic to FB and tensor frameworks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from aeris.contracts.agent import TrainableAgent
from aeris.contracts.dataset import BatchProvider
from aeris.contracts.recorder import RunRecorder
from aeris.domain.results import MetricValue, TrainingMetricRecord
from aeris.domain.validation import require_int


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    update_count: int
    last_metrics: tuple[MetricValue, ...]


def train_updates(
    agent: TrainableAgent[Any], batches: BatchProvider[Any], *, updates: int,
    run_id: str, recorder: RunRecorder | None = None,
    start_step: int = 0, log_every: int = 1,
) -> TrainingSummary:
    """Execute gradient updates with externally prepared batches.

    The runner never needs to know the learner's network layout, loss or RNG.
    `start_step` is the number of prior updates in a resumed run.
    """
    require_int("updates", updates)
    require_int("start_step", start_step)
    require_int("log_every", log_every, minimum=1)
    last: tuple[MetricValue, ...] = ()
    for ix in range(1, updates + 1):
        last = tuple(agent.update(batches.next_batch()))
        if recorder is not None and (ix % log_every == 0 or ix == updates):
            recorder.write_training_metrics(TrainingMetricRecord(
                run_id=run_id, step_axis="gradient_updates", step=start_step + ix, metrics=last))
    return TrainingSummary(update_count=updates, last_metrics=last)
