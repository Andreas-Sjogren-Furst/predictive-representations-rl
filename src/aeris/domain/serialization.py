"""Conversion between domain records and portable JSON-compatible data."""

from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from math import isfinite
from types import MappingProxyType, UnionType
from typing import Any, Literal, Self, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")


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


def from_json_data(cls: type[T], data: Any) -> T:
    """Build a domain dataclass from plain JSON data, checking it against the type hints.

    Unknown or missing fields are errors, so a malformed artifact fails while
    reading instead of deep inside an analysis. Fields annotated ``Any`` are
    passed through unchanged; the record's own ``__post_init__`` still runs.
    """
    return _decode(cls, data, cls.__name__)


def _decode(tp: Any, value: Any, path: str) -> Any:
    if tp is Any:
        return value
    origin = get_origin(tp)
    args = get_args(tp)
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for option in (a for a in args if a is not type(None)):
            try:
                return _decode(option, value, path)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        raise ValueError("; ".join(errors))
    if origin is Literal:
        if value not in args:
            raise ValueError(f"{path}: expected one of {args!r}, got {value!r}")
        return value
    if isinstance(tp, type) and is_dataclass(tp):
        return _decode_dataclass(tp, value, path)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path}: expected a list, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ValueError(f"{path}: expected {len(args)} items, got {len(value)}")
        return tuple(
            _decode(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value, strict=True))
        )
    if origin in (Mapping, dict):
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}: expected an object, got {type(value).__name__}")
        value_type = args[1] if args else Any
        return {k: _decode(value_type, v, f"{path}.{k}") for k, v in value.items()}
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp in (str, bool, int):
        if type(value) is not tp:
            raise ValueError(f"{path}: expected {tp.__name__}, got {value!r}")
        return value
    raise TypeError(f"{path}: unsupported field type {tp!r}")


def _decode_dataclass(cls: type[Any], value: Any, path: str) -> Any:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected an object, got {type(value).__name__}")
    hints = get_type_hints(cls)
    known = {f.name: f for f in fields(cls) if f.init}
    unknown = sorted(set(value) - set(known))
    if unknown:
        raise ValueError(f"{path}: unknown field(s) {', '.join(unknown)}")
    missing = [
        name
        for name, f in known.items()
        if name not in value and f.default is MISSING and f.default_factory is MISSING
    ]
    if missing:
        raise ValueError(f"{path}: missing field(s) {', '.join(missing)}")
    kwargs = {name: _decode(hints[name], v, f"{path}.{name}") for name, v in value.items()}
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


class JsonRecord:
    """Mixin for domain records that round-trip through JSON artifacts."""

    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        return to_json_data(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return from_json_data(cls, data)
