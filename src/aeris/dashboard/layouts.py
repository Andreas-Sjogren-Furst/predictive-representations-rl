"""Dashboards as data: which panels, with which options, in which order and width."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aeris.domain.serialization import JsonRecord
from aeris.domain.validation import require_identifier, require_int


@dataclass(frozen=True, slots=True, kw_only=True)
class PanelSpec(JsonRecord):
    panel: str  # a name registered in aeris.dashboard.panels.PANELS
    width: int = 1  # grid columns spanned
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_identifier("panel", self.panel)
        require_int("width", self.width, minimum=1)


@dataclass(frozen=True, slots=True, kw_only=True)
class Layout(JsonRecord):
    name: str
    columns: int = 2
    panels: tuple[PanelSpec, ...] = ()

    def __post_init__(self) -> None:
        require_identifier("name", self.name)
        require_int("columns", self.columns, minimum=1)
        object.__setattr__(self, "panels", tuple(self.panels))

    @property
    def file_name(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_") + ".json"


BUILT_IN = (
    Layout(
        name="Run overview",
        panels=(
            PanelSpec(panel="loss_curves", options={"metrics": ["fb_loss", "fb_regularization"]}),
            PanelSpec(panel="evaluation"),
            PanelSpec(panel="policy_map"),
            PanelSpec(panel="trajectory"),
            PanelSpec(panel="goal_scores_map"),
            PanelSpec(panel="report"),
            PanelSpec(panel="run_overview", width=2),
        ),
    ),
    Layout(
        name="Zero-shot suite",
        panels=(
            PanelSpec(panel="suite_scores", width=2),
            PanelSpec(panel="suite_distribution", width=2),
            PanelSpec(panel="task_policy", options={"task": "room_top_left"}),
            PanelSpec(panel="task_policy", options={"task": "random_dense_0"}),
            PanelSpec(panel="goal_scores_map"),
            PanelSpec(panel="task_policy", options={"task": "distance_9_9"}),
            PanelSpec(panel="suite_tasks", width=2),
        ),
    ),
    Layout(
        name="Behaviour",
        panels=(
            PanelSpec(panel="rollout"),
            PanelSpec(panel="fate_map"),
            PanelSpec(panel="rollout", options={"task": "room_bottom_right", "start": "1,1"}),
            PanelSpec(panel="fate_map", options={"task": "room_bottom_right"}),
            PanelSpec(panel="fate_map", options={"colour": "start score"}),
            PanelSpec(panel="fate_map", options={"task": "random_dense_0"}),
        ),
    ),
    Layout(
        name="Representations",
        panels=(
            PanelSpec(panel="projection", options={"color": "room"}),
            PanelSpec(panel="similarity"),
            PanelSpec(panel="occupancy_maps", width=2),
            PanelSpec(panel="value_map", width=2),
            PanelSpec(panel="projection", options={"dimensions": 3, "color": "distance to goal"}),
            PanelSpec(panel="b_spectrum"),
        ),
    ),
    Layout(
        name="Compare runs",
        panels=(
            PanelSpec(panel="group_summary", width=2),
            PanelSpec(panel="run_comparison", width=2),
            PanelSpec(
                panel="loss_curves",
                options={"metrics": ["fb_loss"], "smoothing": 5, "group_by": "name"},
            ),
            PanelSpec(panel="evaluation"),
            PanelSpec(panel="suite_scores", width=2),
        ),
    ),
)


def load_layouts(directory: str | Path) -> dict[str, Layout]:
    """Saved layouts by name; built-in layouts fill in names that are not saved."""
    layouts = {layout.name: layout for layout in BUILT_IN}
    directory = Path(directory)
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            layout = Layout.from_dict(json.loads(path.read_text(encoding="utf-8")))
            layouts[layout.name] = layout
    return layouts


def save_layout(directory: str | Path, layout: Layout) -> Path:
    path = Path(directory) / layout.file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(layout.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
