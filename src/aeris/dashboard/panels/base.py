"""The panel contract: typed run data in, a figure or table out.

A panel never imports Streamlit and never reads files. It declares its options
as data (``Choice``, ``MultiChoice``, ``Number``, ``Toggle``) so any shell can
draw the right widgets, and it returns a Plotly figure, a ``Table``, a row of
``Stats`` or a ``Message``. A new visualization is a new ``Panel`` subclass registered in
``panels/__init__.py``; no shell code changes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeAlias, Union

import plotly.graph_objects as go

from aeris.dashboard.data import RunData

Scope = Literal["run", "runs"]


@dataclass(frozen=True)
class Choice:
    key: str
    label: str
    choices: tuple[Any, ...]
    default: Any = None


@dataclass(frozen=True)
class MultiChoice:
    key: str
    label: str
    choices: tuple[Any, ...]
    default: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Number:
    key: str
    label: str
    default: int
    minimum: int = 0
    maximum: int = 1_000_000


@dataclass(frozen=True)
class Toggle:
    key: str
    label: str
    default: bool = False


Option = Choice | MultiChoice | Number | Toggle

# Series colours in fixed order (never cycled): slot n always means the n-th series.
CATEGORICAL = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
OVERFLOW = "#a3a29c"  # every series past the eighth


def series_colour(index: int) -> str:
    return CATEGORICAL[index] if index < len(CATEGORICAL) else OVERFLOW


@dataclass(frozen=True)
class Table:
    rows: Sequence[Mapping[str, Any]]
    caption: str = ""


@dataclass(frozen=True)
class Stat:
    """One headline number: a label, the value as text, and an optional muted line."""

    label: str
    value: str
    detail: str = ""
    delta: str | None = None  # a change, signed, e.g. "+76 pts vs initial"


@dataclass(frozen=True)
class Stats:
    stats: Sequence[Stat]
    caption: str = ""


@dataclass(frozen=True)
class Message:
    text: str
    kind: Literal["info", "warning"] = "info"


Output: TypeAlias = Union[go.Figure, Table, Stats, Message]  # noqa: UP007 (go.Figure is untyped)


@dataclass(frozen=True)
class Panel:
    """Base class; subclasses set ``name``, ``title``, ``scope`` and implement ``render``."""

    name: ClassVar[str]
    title: ClassVar[str]
    scope: ClassVar[Scope]
    description: ClassVar[str] = ""

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return ()

    def missing(self, runs: Sequence[RunData]) -> str | None:
        """Why the panel cannot render for these runs (with the command to fix it), or None."""
        return None

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        raise NotImplementedError

    def resolve_options(self, runs: Sequence[RunData], given: Mapping[str, Any]) -> dict[str, Any]:
        """Saved option values, falling back to defaults when absent or no longer valid."""
        resolved: dict[str, Any] = {}
        for option in self.options(runs):
            value = given.get(option.key, option.default)
            if isinstance(option, Choice) and value not in option.choices:
                value = (
                    option.default if option.default in option.choices else _first(option.choices)
                )
            elif isinstance(option, MultiChoice):
                value = tuple(v for v in (value or ()) if v in option.choices) or tuple(
                    option.default
                )
            resolved[option.key] = value
        return resolved


def _first(choices: tuple[Any, ...]) -> Any:
    return choices[0] if choices else None


@dataclass(frozen=True)
class FigureStyle:
    """Shared sizing so panels look consistent in the grid."""

    height: int = 420
    margin: dict[str, int] = field(default_factory=lambda: {"l": 40, "r": 20, "t": 50, "b": 40})

    def apply(self, fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(title=title, height=self.height, margin=self.margin, legend_title_text="")
        return fig


STYLE = FigureStyle()
