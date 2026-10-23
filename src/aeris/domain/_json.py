from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any


def freeze_settings(settings: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping")
    
    return _freeze_value(settings)

def _freeze_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("configuration numbers must be finite")
        return value
    
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("configuration keys must be strings")
        return MappingProxyType({k: _freeze_value(v) for k, v in value.items()})
    
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(v) for v in value)
    
    raise TypeError(f"unsupported configuration value: {type(value).__name__}")

def to_plain_data(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_plain_data(getattr(value, f.name)) for f in fields(value)}
    
    if isinstance(value, Mapping):
        return {k: to_plain_data(v) for k, v in value.items()}
    
    if isinstance(value, (tuple, list)):
        return [to_plain_data(v) for v in value]
    
    return value

def require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")

def require_seed(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
