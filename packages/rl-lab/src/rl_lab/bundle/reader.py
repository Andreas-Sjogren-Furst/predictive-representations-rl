"""Read and validate bundles and diagnostic datasets.

Validation collects every problem it finds (errors block use, warnings do not) instead of stopping at the first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from rl_lab.bundle.io import DATASET_SPEC, INPUTS, MANIFEST, SAMPLES, read_npz, read_yaml, resolve_inside
from rl_lab.bundle.writer import dataset_hash
from rl_lab.contracts import DatasetSpec, Manifest


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class Dataset:
    spec: DatasetSpec
    samples: pd.DataFrame
    inputs: dict[str, np.ndarray] | None
    directory: Path


def load_manifest(bundle: Path) -> Manifest:
    return Manifest.model_validate(read_yaml(bundle / MANIFEST))


def load_dataset(directory: Path) -> Dataset:
    spec = DatasetSpec.model_validate(read_yaml(directory / DATASET_SPEC))
    samples = pd.read_parquet(directory / SAMPLES)
    inputs = read_npz(directory / INPUTS) if (directory / INPUTS).exists() else None
    return Dataset(spec, samples, inputs, directory)


def load_view(bundle: Path, manifest: Manifest, view_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(sample_ids, embeddings) of one view."""
    arrays = read_npz(resolve_inside(bundle, manifest.view(view_id).path))
    return arrays["sample_ids"].astype(str), arrays["embeddings"]


def validate_dataset(directory: Path) -> Report:
    report = Report()
    try:
        dataset = load_dataset(directory)
    except (OSError, ValueError, ValidationError) as error:
        report.errors.append(f"{directory}: cannot load dataset: {error}")
        return report

    samples, spec = dataset.samples, dataset.spec
    if "sample_id" not in samples.columns:
        report.errors.append("samples.parquet has no sample_id column")
        return report
    if samples["sample_id"].duplicated().any():
        report.errors.append("duplicate sample_id values")
    if len(samples) != spec.construction.num_samples:
        report.errors.append(f"{len(samples)} samples, dataset.yaml says {spec.construction.num_samples}")
    for label in spec.labels:
        if label not in samples.columns:
            report.errors.append(f"label {label!r} is declared but has no column")
    if spec.grouping and spec.grouping.column not in samples.columns:
        report.errors.append(f"grouping column {spec.grouping.column!r} is missing")
    actual = dataset_hash(samples, dataset.inputs)
    if actual != spec.hash:
        report.errors.append(f"content hash mismatch: dataset.yaml {spec.hash}, files {actual}")
    return report


def validate_bundle(bundle: Path, datasets_root: Path) -> Report:
    """Check a bundle against the contract and the catalog's datasets."""
    report = Report()
    try:
        manifest = load_manifest(bundle)
    except (OSError, ValueError, ValidationError) as error:
        report.errors.append(f"{bundle}: invalid manifest: {error}")
        return report

    sample_ids_by_dataset: dict[str, set[str]] = {}
    for view in manifest.views:
        where = f"view {view.id!r}"
        dataset_dir = datasets_root / view.dataset.id
        if view.dataset.id not in sample_ids_by_dataset:
            if not (dataset_dir / DATASET_SPEC).exists():
                report.errors.append(f"{where}: dataset {view.dataset.id!r} not found in {datasets_root}")
                sample_ids_by_dataset[view.dataset.id] = set()
                continue
            dataset = load_dataset(dataset_dir)
            if dataset.spec.hash != view.dataset.hash:
                report.errors.append(f"{where}: dataset {view.dataset.id!r} has changed since export "
                                     f"({view.dataset.hash} != {dataset.spec.hash})")
            sample_ids_by_dataset[view.dataset.id] = set(dataset.samples["sample_id"].astype(str))

        try:
            arrays = read_npz(resolve_inside(bundle, view.path))
        except (OSError, ValueError) as error:
            report.errors.append(f"{where}: cannot read {view.path}: {error}")
            continue
        if set(arrays) != {"sample_ids", "embeddings"}:
            report.errors.append(f"{where}: expected arrays sample_ids and embeddings, found {sorted(arrays)}")
            continue
        sample_ids, embeddings = arrays["sample_ids"].astype(str), arrays["embeddings"]
        if embeddings.ndim != 2 or len(embeddings) != len(sample_ids):
            report.errors.append(f"{where}: embeddings shape {embeddings.shape} does not match {len(sample_ids)} sample ids")
        elif not np.isfinite(embeddings).all():
            report.errors.append(f"{where}: embeddings contain non-finite values")
        if len(set(sample_ids)) != len(sample_ids):
            report.errors.append(f"{where}: duplicate sample ids")
        unknown = set(sample_ids) - sample_ids_by_dataset.get(view.dataset.id, set())
        if unknown and sample_ids_by_dataset.get(view.dataset.id):
            report.errors.append(f"{where}: {len(unknown)} sample ids are not in dataset {view.dataset.id!r}")

    for artifact in manifest.artifacts:
        try:
            if not resolve_inside(bundle, artifact.path).exists():
                report.errors.append(f"artifact {artifact.path!r} is missing")
        except ValueError as error:
            report.errors.append(f"artifact: {error}")

    if manifest.experiment.environment.task_definition is None:
        report.warnings.append("no task-definition fingerprint: comparisons cannot check that rewards match")
    trained = {view.id for view in manifest.views if view.role == "trained"}
    controlled = {view.control_of for view in manifest.views if view.role == "control"}
    for view_id in sorted(trained - controlled):
        report.warnings.append(f"view {view_id!r} has no untrained control")
    return report
