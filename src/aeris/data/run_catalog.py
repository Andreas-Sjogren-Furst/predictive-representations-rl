"""Read-only discovery and inspection of existing AERIS run artifacts.

A catalog is derived from saved manifests, not a separate database. These
functions deliberately do not import agents or analysis implementations.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any


def _read_object(path: Path) -> dict[str, Any]:
    """Load one JSON object and report which artifact is malformed."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"no manifest.json found in {run_dir}")
    manifest = _read_object(path)
    if not isinstance(manifest.get("run_id"), str):
        raise ValueError(f"missing run_id in {path}")
    if not isinstance(manifest.get("experiment"), dict):
        raise ValueError(f"missing experiment in {path}")
    return manifest


def _summary(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    experiment = manifest["experiment"]
    agent = experiment.get("agent") or {}
    return {
        "directory": str(run_dir),
        "run_id": manifest["run_id"],
        "status": manifest.get("status", "unknown"),
        "experiment": experiment.get("name", "unknown"),
        "agent": agent.get("name", "unknown") if isinstance(agent, dict) else "unknown",
        "seed": experiment.get("seed"),
    }


def list_runs(root: str | Path) -> list[dict[str, Any]]:
    """Discover immediate child directories with AERIS manifests."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"runs directory does not exist: {root}")
    runs = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "manifest.json").is_file():
            runs.append(_summary(child, _manifest(child)))
    return runs


def resolve_run(reference: str | Path, *, root: str | Path = "runs") -> Path:
    """Accept an explicit directory, directory name, or unique saved run_id."""
    reference = Path(reference)
    candidates = (reference, Path(root) / reference)
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate

    # A manifest's run_id may differ from its directory name. Resolve those
    # identifiers only when an unambiguous match exists.
    root_path = Path(root)
    if root_path.is_dir() and len(reference.parts) == 1:
        matches = [
            Path(item["directory"])
            for item in list_runs(root_path)
            if item["run_id"] == str(reference)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"multiple runs have id {reference!s}; use a directory path")
    raise FileNotFoundError(f"run not found: {reference} (root: {root_path})")


def _recent_jsonl(path: Path, *, limit: int = 5) -> tuple[int, list[dict[str, Any]]]:
    """Read event count and only a small tail, without loading a long run into RAM."""
    if not path.exists():
        return 0, []
    tail: deque[dict[str, Any]] = deque(maxlen=limit)
    count = 0
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"expected JSON object in {path}:{lineno}")
            count += 1
            tail.append(obj)
    return count, list(tail)


def inspect_run(run_dir: str | Path) -> dict[str, Any]:
    """Inspect a run without importing an algorithm or loading checkpoints."""
    run_dir = Path(run_dir)
    manifest = _manifest(run_dir)
    num_train, recent_train = _recent_jsonl(run_dir / "training_metrics.jsonl", limit=1)
    num_eval, recent_eval = _recent_jsonl(run_dir / "evaluations.jsonl")
    return {
        **_summary(run_dir, manifest),
        "manifest": manifest,
        "training_record_count": num_train,
        "last_training_record": recent_train[-1] if recent_train else None,
        "evaluation_record_count": num_eval,
        "recent_evaluations": recent_eval,
    }
