"""Local self-describing run recorder writing JSON and JSON Lines files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from aeris.domain.results import EvaluationRecord, RunManifest, TrainingMetricRecord
from aeris.domain.serialization import to_json_data
from aeris.domain.trajectory import EvaluationTrajectory


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read every nonblank line of a JSON Lines file."""
    lines = Path(path).read_text(encoding="utf8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


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

    def write_training_metrics(self, record: TrainingMetricRecord) -> None:
        self._append("training_metrics.jsonl", record.to_dict())

    def write_evaluation(self, record: EvaluationRecord) -> None:
        self._append("evaluations.jsonl", record.to_dict())

    def write_trajectory(self, record: EvaluationTrajectory) -> None:
        """Append a compact diagnostic episode (not meant for image observations)."""
        self._append("trajectories.jsonl", to_json_data(record))

    def _append(self, filename: str, obj: dict[str, Any]) -> None:
        # Serialize before opening so a rejected record never leaves a partial line.
        line = json.dumps(obj, sort_keys=True, allow_nan=False) + "\n"
        with (self.root / filename).open("a", encoding="utf8") as stream:
            stream.write(line)
