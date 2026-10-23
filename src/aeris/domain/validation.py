"""Argument checks shared by domain records and runtime components."""

from math import isfinite
from typing import Any


def require_identifier(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def require_int(name: str, value: Any, *, minimum: int = 0) -> None:
    """Require a plain integer (not bool) of at least ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def require_finite_number(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a scalar number")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
