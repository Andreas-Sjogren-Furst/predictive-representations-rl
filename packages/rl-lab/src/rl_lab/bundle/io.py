"""Low-level, safe file access and content hashing for bundles and datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MANIFEST = "manifest.yaml"
DATASET_SPEC = "dataset.yaml"
SAMPLES = "samples.parquet"
INPUTS = "inputs.npz"


def content_hash(columns: dict[str, np.ndarray]) -> str:
    """sha256 over the named arrays' names, dtypes, shapes and values, independent of file format and library
    versions (unlike hashing Parquet bytes, which carry writer metadata)."""
    digest = hashlib.sha256()
    for name in sorted(columns):
        values = np.asarray(columns[name])
        if values.dtype.kind in "OUS":
            values = np.array([str(v) for v in values.ravel()], dtype=object).reshape(values.shape)
            payload = "\0".join(values.ravel()).encode()
            dtype = "str"
        else:
            payload = np.ascontiguousarray(values).tobytes()
            dtype = values.dtype.str
        digest.update(f"{name}|{dtype}|{values.shape}|".encode())
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    return content_hash({column: frame[column].to_numpy() for column in frame.columns})


def read_npz(path: Path) -> dict[str, np.ndarray]:
    """Arrays of an .npz file. Pickled objects are refused."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    return data


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def resolve_inside(root: Path, relative: str) -> Path:
    """`root / relative`, refusing absolute paths, `..` escapes and symlinks."""
    if Path(relative).is_absolute():
        raise ValueError(f"path {relative!r} must be relative to the bundle")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path {relative!r} escapes the bundle")
    if (root / relative).is_symlink():
        raise ValueError(f"path {relative!r} is a symlink")
    return path
