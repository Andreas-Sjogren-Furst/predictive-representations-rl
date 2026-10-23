"""Conversion between domain records and portable JSON-compatible data."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any


def freeze_settings(settings: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a read-only deep copy of JSON-compatible configuration."""
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping")
    return _freeze(settings)


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("configuration numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("configuration keys must be strings")
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    raise TypeError(f"unsupported configuration value: {type(value).__name__}")


def to_json_data(value: Any) -> Any:
    """Convert dataclasses, mappings, sequences and arrays to plain JSON data.

    Rejects nonfinite numbers and arbitrary objects rather than stringifying
    them. Objects exposing ``tolist()`` (NumPy arrays, CPU tensors) are
    converted; GPU tensors must be moved to the CPU by the caller.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_json_data(getattr(value, f.name)) for f in fields(value)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON data must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("JSON mapping keys must be strings")
        return {k: to_json_data(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_json_data(v) for v in value]
    if callable(getattr(value, "tolist", None)):
        return to_json_data(value.tolist())
    raise TypeError(f"cannot convert {type(value).__name__} to JSON data")
