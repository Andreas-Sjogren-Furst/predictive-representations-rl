"""Local self-describing recorder with optional evaluation trajectory artifacts."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import json
import math
import os
from pathlib import Path

from aeris.domain.results import EvaluationRecord, RunManifest, TrainingMetricRecord
from aeris.domain.trajectory import EvaluationTrajectory


def _json_observation(value):
    """Convert small numeric/structured runtime payloads to portable JSON.

    Strictly reject nonfinite values and arbitrary Python objects rather than
    stringifying them. Not intended for high-volume image or latent logging.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_observation(getattr(value, f.name)) for f in fields(value)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("trajectory data must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("trajectory mapping keys must be strings")
        return {k: _json_observation(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_observation(v) for v in value]
    # NumPy and CPU torch tensors expose tolist(). GPU tensors should be
    # explicitly transferred by the caller to avoid hidden costly copies.
    if hasattr(value, "tolist") and callable(value.tolist):
        return _json_observation(value.tolist())
    raise TypeError(f"cannot JSON-serialize trajectory value of type {type(value).__name__}")


class JsonlRunRecorder:
    """One run per directory, atomic manifest writes, append-only metrics/traces."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, manifest: RunManifest) -> None:
        destination = self.root / "manifest.json"
        temporary = self.root / "manifest.json.tmp"
        with temporary.open("w", encoding="utf8") as stream:
            json.dump(manifest.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)

    def _append(self, filename: str, obj: dict) -> None:
        with (self.root / filename).open("a", encoding="utf8") as stream:
            stream.write(json.dumps(obj, sort_keys=True, allow_nan=False) + "\n")

    def write_training_metrics(self, record: TrainingMetricRecord) -> None:
        self._append("training_metrics.jsonl", record.to_dict())

    def write_evaluation(self, record: EvaluationRecord) -> None:
        self._append("evaluations.jsonl", record.to_dict())

    def write_trajectory(self, record: EvaluationTrajectory) -> None:
        """Append a compact diagnostic episode; no agent or environment import."""
        obj = _json_observation(record)  # Convert/validate before file append.
        self._append("trajectories.jsonl", obj)
