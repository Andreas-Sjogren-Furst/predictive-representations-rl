"""Export harness runs as rl_lab bundles (`prl export`).

catalog/                                    # $PRL_CATALOG, default <project>/catalog
├── datasets/<env>-probes-<name>/           # the shared probe set: samples.parquet, inputs.npz, dataset.yaml
└── experiments/<experiment id>/            # one run: manifest.yaml, representations/, metrics/, provenance/

Bundles are derived data: re-exporting a run replaces its bundle. A dataset is never overwritten with different
content, since views exported against it would silently stop matching.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rl_lab.bundle.writer import write_dataset, write_manifest, write_table, write_view
from rl_lab.contracts import SCHEMA_VERSION, DatasetSpec, Manifest

from predictive_representations_rl.envs.base import task_definition_fingerprint
from predictive_representations_rl.probes.probe_set import ProbeSet

CATALOG_VARIABLE = "PRL_CATALOG"
EXPORTER = {"name": "prl-export", "version": "0.1.0"}
# Probe-set metadata columns that identify a sample rather than describe it.
SAMPLE_COLUMNS = {"source_index": "sample_id", "episode": "episode_id", "step": "timestep"}
UNTRAINED_SUFFIX = "_untrained"
MAX_CLASSES = 20


class ExportError(RuntimeError):
    pass


def catalog_dir(root: Path) -> Path:
    return Path(os.environ.get(CATALOG_VARIABLE, root / "catalog")).expanduser()


def dataset_id(env_name: str, probe_name: str) -> str:
    return f"{env_name}-probes-{probe_name}"


def experiment_id(run) -> str:
    parts = [run.config.algo, run.config.env, run.config.mode, run.config.task, run.config.variant, f"s{run.seed}"]
    return "--".join(part for part in parts if part)


def experiment_group(run) -> str:
    parts = [run.config.algo, run.config.env, run.config.mode, run.config.task, run.config.variant]
    return "/".join(part for part in parts if part)


def sample_id(index: int) -> str:
    return f"{int(index):07d}"


def export_dataset(run, probe_name: str, catalog: Path) -> DatasetSpec:
    """Write the run's probe set as a shared diagnostic dataset (or confirm the existing one is identical)."""
    from predictive_representations_rl.runner import probe_set_path

    path = probe_set_path(run.root, run.env, probe_name)
    probe_set = ProbeSet.load(path)
    params = json.loads(path.with_suffix(".json").read_text())
    meta = probe_set.metadata
    task_definition = task_definition_fingerprint(run.env, run.root)

    columns: dict[str, Any] = {
        "sample_id": [sample_id(i) for i in meta["source_index"]],
        "episode_id": [f"{int(e):04d}" for e in meta["episode"]],
        "timestep": meta["step"].astype(np.int64),
    }
    labels: dict[str, dict[str, Any]] = {}
    for name, values in meta.items():
        if name in SAMPLE_COLUMNS:
            continue
        columns[name] = values
        if name.startswith("reward_"):
            task = name.removeprefix("reward_")
            definition = f"per-step reward of {task}" + (f" under {task_definition['fingerprint']}" if task_definition else "")
            labels[name] = {"kind": "reward", "task": task, "definition": definition}
        elif np.issubdtype(values.dtype, np.integer) and len(np.unique(values)) <= MAX_CLASSES:
            labels[name] = {"kind": "categorical", "classes": int(len(np.unique(values)))}
        else:
            labels[name] = {"kind": "continuous"}
    samples = pd.DataFrame(columns)

    ids = samples["sample_id"].to_numpy().astype(str)
    inputs = {
        "sample_ids": ids,
        "observations": probe_set.final_observations,
        "actions": probe_set.final_actions,
        "window_observations": probe_set.observations,
        "window_actions": probe_set.actions,
    }
    coverage = {name: {str(k): int(v) for k, v in zip(*np.unique(meta[name], return_counts=True))}
                for name, label in labels.items() if label["kind"] == "categorical"}
    spec = DatasetSpec.model_validate({
        "id": dataset_id(run.env.name, probe_name),
        "hash": "",
        "source": {"kind": "held_out_dataset", "id": Path(params["source"]).stem,
                   "note": "held-out split, never used for training"},
        "construction": {"method": "random_windows", "num_samples": int(params["num"]),
                         "window_length": int(params["length"]), "seed": int(params["seed"]),
                         "constraints": ["window inside one episode", "excludes each episode's placeholder final step"]},
        "labels": labels,
        "grouping": {"column": "episode_id"},
        "coverage": coverage,
    })
    try:
        return write_dataset(catalog / "datasets" / spec.id, spec, samples, inputs)
    except ValueError as error:
        raise ExportError(str(error)) from error


def export_run(run, probe_name: str = "default", catalog: Path | None = None) -> Path:
    """Export one extracted run as a bundle. Returns the bundle directory."""
    catalog = catalog or catalog_dir(run.root)
    record_path = run.run_dir / "run.json"
    if not record_path.exists():
        raise ExportError(f"{run.run_dir} has no run.json")
    record = json.loads(record_path.read_text())
    if record.get("status") != "completed":
        raise ExportError(f"{run.run_dir} is {record.get('status')}, not completed")
    extraction = record.get("extractions", {}).get(probe_name)
    if extraction is None:
        raise ExportError(f"{run.run_dir} has no extraction for probe set {probe_name!r}; run `prl extract` first")

    dataset = export_dataset(run, probe_name, catalog)
    probe_dir = Path(extraction["dir"])
    info = json.loads((probe_dir / "extract_info.json").read_text())
    info["window_length"] = dataset.construction.window_length
    checkpoint_step, checkpoint_unit = run.adapter.checkpoint_step(run, Path(extraction["checkpoint"]))

    bundle = catalog / "experiments" / experiment_id(run)
    if bundle.exists():
        shutil.rmtree(bundle)

    ids = np.array([sample_id(i) for i in ProbeSet.load(Path(extraction["probe_set"])).metadata["source_index"]])
    views, artifacts = [], []
    for name in info["representations"]:
        with np.load(probe_dir / f"{name}.npz", allow_pickle=False) as data:
            values, tasks = data["values"], (data["tasks"] if "tasks" in data.files else None)
        if values.ndim != 2 or len(values) != len(ids):
            # Not one row per sample (e.g. FB's per-task z): kept as an artifact, not a view.
            path = f"artifacts/{name}.npz"
            np.savez_compressed(_ensure(bundle / path), values=values, **({"tasks": tasks.astype(str)} if tasks is not None else {}))
            artifacts.append({"kind": "task_latents", "path": path, "description": f"{name}: {list(values.shape)}"})
            continue

        is_control = name.endswith(UNTRAINED_SUFFIX)
        base = name.removesuffix(UNTRAINED_SUFFIX)
        view_id = f"{base}__untrained" if is_control else name
        step = 0 if is_control else checkpoint_step  # untrained controls are the network at initialisation
        path = f"representations/step_{step}/{view_id}.npz"
        write_view(bundle, path, ids, values.astype(np.float32))
        views.append({
            "id": view_id,
            "role": "control" if is_control else "trained",
            **({"control_kind": "untrained", "control_of": base} if is_control else {}),
            "checkpoint_step": step,
            "checkpoint_unit": checkpoint_unit,
            "dataset": {"id": dataset.id, "hash": dataset.hash},
            "path": path,
            "protocol": run.adapter.view_protocol(run, name, info),
        })

    training_rows, evaluation_rows = run.adapter.native_metrics(run)
    for kind, rows, path in (("training_metrics", training_rows, "metrics/training.parquet"),
                             ("evaluation_metrics", evaluation_rows, "metrics/evaluation.parquet")):
        if rows:
            write_table(bundle, path, pd.DataFrame(rows))
            artifacts.append({"kind": kind, "path": path})

    for source in (record_path, probe_dir / "extract_info.json"):
        shutil.copy2(source, _ensure(bundle / "provenance" / source.name))
        artifacts.append({"kind": "provenance", "path": f"provenance/{source.name}"})

    task_definition = record.get("env", {}).get("task_definition")
    note = None
    if task_definition is None:
        task_definition, note = task_definition_fingerprint(run.env, run.root), "computed at export time from the current files"
    data_kind = {"self_collected": "self_collected", "dataset": "dataset", "replay": "replay"}[record["algorithm"]["data_requirement"]]
    manifest = Manifest.model_validate({
        "schema_version": SCHEMA_VERSION,
        "exporter": {**EXPORTER, "source_revision": record.get("git", {}).get(".")},
        "experiment": {
            "id": experiment_id(run),
            "group": experiment_group(run),
            "seed": run.seed,
            "algorithm": {
                "name": run.adapter.spec.name,
                "revision": _algorithm_revision(run, record),
                "configuration": {**run.config.overrides, **{f"budget.{k}": v for k, v in run.config.budget.items()}},
            },
            "environment": {
                "id": f"{run.env.suite}/{run.env.name}",
                "tasks": list(run.env.tasks),
                **({"task_definition": {**task_definition, **({"note": note} if note else {})}} if task_definition else {}),
            },
            "training": {
                "mode": run.config.mode,
                "objective": "task_reward" if run.adapter.spec.task_specific else "reward_free",
                "trained_task": run.config.task,
                "data": {"kind": data_kind, "id": _data_id(run), "transitions": _transitions(run)},
                "budget": run.adapter.budget(run),
            },
        },
        "views": views,
        "artifacts": artifacts,
    })
    write_manifest(bundle, manifest)
    return bundle


def _algorithm_revision(run, record: dict[str, Any]) -> str | None:
    repo = next((path for path in record.get("git", {}) if path != "." and path in record["command"]["cwd"]), None)
    return record["git"].get(repo) if repo else None


def _data_id(run) -> str | None:
    return f"{run.config.dataset}/{run.env.name}" if run.config.dataset else None


def _transitions(run) -> int | None:
    if not run.config.dataset:
        return None
    path = Path(run.env.datasets[run.config.dataset].files[0]).expanduser()
    if not path.exists():
        return None
    import h5py

    with h5py.File(path, "r") as file:
        return int(file["observations"].shape[0])


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
