"""Read-only discovery and inspection of existing AERIS run directories.

A catalog is derived from saved manifests, not a separate database. These
functions deliberately do not import agents or analysis implementations.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from aeris.data.run_directory import RunDirectory
from aeris.domain.results import EvaluationRecord, RunManifest, RunStatus, TrainingMetricRecord
from aeris.domain.serialization import JsonRecord

T = TypeVar("T")


@dataclass(frozen=True, slots=True, kw_only=True)
class RunSummary(JsonRecord):
    """One line of a run listing."""

    directory: str
    run_id: str
    status: RunStatus
    experiment: str
    agent: str
    seed: int

    @classmethod
    def of(cls, run: RunDirectory, manifest: RunManifest) -> RunSummary:
        experiment = manifest.experiment
        return cls(
            directory=str(run.root),
            run_id=manifest.run_id,
            status=manifest.status,
            experiment=experiment.name,
            agent=experiment.agent.name,
            seed=experiment.seed,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RunDetails(RunSummary):
    """A summary plus the manifest and the most recent recorded metrics."""

    manifest: RunManifest
    training_record_count: int
    last_training_record: TrainingMetricRecord | None
    evaluation_record_count: int
    recent_evaluations: tuple[EvaluationRecord, ...]


def list_runs(root: str | Path) -> list[RunSummary]:
    """Summarize immediate child directories that contain a run manifest."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"runs directory does not exist: {root}")
    runs = (RunDirectory(child) for child in sorted(root.iterdir()) if child.is_dir())
    return [RunSummary.of(run, run.read_manifest()) for run in runs if run.exists()]


def resolve_run(reference: str | Path, *, root: str | Path = "runs") -> RunDirectory:
    """Accept an explicit directory, a directory name under ``root``, or a unique run_id."""
    reference = Path(reference)
    for candidate in (reference, Path(root) / reference):
        if RunDirectory(candidate).exists():
            return RunDirectory(candidate)

    # A manifest's run_id may differ from its directory name. Resolve those
    # identifiers only when an unambiguous match exists.
    root_path = Path(root)
    if root_path.is_dir() and len(reference.parts) == 1:
        matches = [run.directory for run in list_runs(root_path) if run.run_id == str(reference)]
        if len(matches) == 1:
            return RunDirectory(matches[0])
        if len(matches) > 1:
            raise ValueError(f"multiple runs have id {reference!s}; use a directory path")
    raise FileNotFoundError(f"run not found: {reference} (root: {root_path})")


def inspect_run(run: RunDirectory, *, recent_evaluations: int = 5) -> RunDetails:
    """Summarize a run without importing an algorithm or loading checkpoints."""
    manifest = run.read_manifest()
    train_count, last_train = _count_and_tail(run.read_training_metrics(), 1)
    eval_count, recent_eval = _count_and_tail(run.read_evaluations(), recent_evaluations)
    return RunDetails(
        **asdict(RunSummary.of(run, manifest)),
        manifest=manifest,
        training_record_count=train_count,
        last_training_record=last_train[-1] if last_train else None,
        evaluation_record_count=eval_count,
        recent_evaluations=tuple(recent_eval),
    )


def _count_and_tail(records: Iterable[T], limit: int) -> tuple[int, list[T]]:
    """Count a record stream and keep only its tail, without loading it into memory."""
    tail: deque[T] = deque(maxlen=limit)
    count = 0
    for record in records:
        count += 1
        tail.append(record)
    return count, list(tail)
