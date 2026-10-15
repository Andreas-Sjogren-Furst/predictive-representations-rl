"""Helpers for exporters: write diagnostic datasets, views, metrics and manifests in the bundle format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rl_lab.bundle.io import DATASET_SPEC, INPUTS, MANIFEST, SAMPLES, content_hash, frame_hash, read_yaml, write_npz, write_yaml
from rl_lab.contracts import DatasetSpec, Manifest


def dataset_hash(samples: pd.DataFrame, inputs: dict[str, np.ndarray] | None = None) -> str:
    """Content hash of a diagnostic dataset: its sample table plus its inputs, if any."""
    parts = {"samples": frame_hash(samples)}
    if inputs:
        parts["inputs"] = content_hash(inputs)
    return content_hash({key: np.array([value]) for key, value in parts.items()})


def write_dataset(
    directory: Path, spec: DatasetSpec, samples: pd.DataFrame, inputs: dict[str, np.ndarray] | None = None
) -> DatasetSpec:
    """Write a dataset (samples.parquet, optional inputs.npz, dataset.yaml). The spec's hash is (re)computed.

    Refuses to overwrite an existing dataset with different content: views computed on the old content would no
    longer match it.
    """
    if "sample_id" not in samples.columns:
        raise ValueError("samples need a sample_id column")
    if samples["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique")
    if inputs is not None and not np.array_equal(inputs.get("sample_ids"), samples["sample_id"].to_numpy().astype(str)):
        raise ValueError("inputs.npz sample_ids must equal the sample table's sample_id column, in order")

    spec = spec.model_copy(update={"hash": dataset_hash(samples, inputs)})
    existing = directory / DATASET_SPEC
    if existing.exists():
        old = DatasetSpec.model_validate(read_yaml(existing))
        if old.hash != spec.hash:
            raise ValueError(f"dataset {spec.id!r} already exists with different content ({old.hash} != {spec.hash})")

    directory.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(directory / SAMPLES, index=False)
    if inputs is not None:
        write_npz(directory / INPUTS, **inputs)
    write_yaml(existing, spec.model_dump(mode="json", exclude_none=True))
    return spec


def write_view(bundle: Path, relative_path: str, sample_ids: np.ndarray, embeddings: np.ndarray) -> None:
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or len(embeddings) != len(sample_ids):
        raise ValueError(f"{relative_path}: embeddings must be (N, D) with N = len(sample_ids)")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{relative_path}: embeddings contain non-finite values")
    write_npz(bundle / relative_path, sample_ids=np.asarray(sample_ids).astype(str), embeddings=embeddings)


def write_table(bundle: Path, relative_path: str, frame: pd.DataFrame) -> None:
    path = bundle / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def write_manifest(bundle: Path, manifest: Manifest) -> None:
    write_yaml(bundle / MANIFEST, manifest.model_dump(mode="json", exclude_none=True))
