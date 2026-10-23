"""AERIS training loop, intentionally agnostic to FB and tensor frameworks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from aeris.contracts.agent import TrainableAgent
from aeris.contracts.dataset import BatchProvider
from aeris.contracts.recorder import RunRecorder
from aeris.domain.results import MetricValue, TrainingMetricRecord


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
    if type(updates) is not int or updates < 0:
        raise ValueError("updates must be a nonnegative integer")
    if type(start_step) is not int or start_step < 0:
        raise ValueError("start_step must be a nonnegative integer")
    if type(log_every) is not int or log_every < 1:
        raise ValueError("log_every must be positive")
    last: tuple[MetricValue, ...] = ()
    for ix in range(1, updates + 1):
        last = tuple(agent.update(batches.next_batch()))
        if recorder is not None and (ix % log_every == 0 or ix == updates):
            recorder.write_training_metrics(TrainingMetricRecord(
                run_id=run_id, step_axis="gradient_updates", step=start_step + ix, metrics=last))
    return TrainingSummary(update_count=updates, last_metrics=last)
