"""AERIS dashboard: the Streamlit shell around the panels. Start with ``aeris dashboard``.

The shell only selects runs, arranges panels from a layout and draws widgets
for the options each panel declares; all plotting lives in ``panels/``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import streamlit as st

from aeris.dashboard.data import RunData, load_run, run_stamp
from aeris.dashboard.layouts import Layout, PanelSpec, load_layouts, save_layout
from aeris.dashboard.panels import PANELS
from aeris.dashboard.panels.base import Choice, Message, MultiChoice, Number, Option, Table, Toggle
from aeris.data.run_catalog import list_runs

RUNS_ROOT = Path(os.environ.get("AERIS_RUNS_ROOT", "runs"))
LAYOUT_DIR = Path(os.environ.get("AERIS_DASHBOARDS", "dashboards"))
RUN_OPTION = "_run"  # shell-level option: which selected run a single-run panel shows


# cache_resource keeps the object itself: RunData is immutable, and its frozen
# mappings cannot be pickled, which st.cache_data would require.
@st.cache_resource(show_spinner=False, max_entries=64)
def _cached_run(root: str, stamp: tuple[tuple[str, int], ...]) -> RunData:
    del stamp  # part of the cache key only: a changed file reloads the run
    return load_run(root)


def get_run(root: Path) -> RunData:
    return _cached_run(str(root), run_stamp(root))


def option_widget(option: Option, value: Any, key: str) -> Any:
    if isinstance(option, Choice):
        index = option.choices.index(value) if value in option.choices else 0
        return (
            st.selectbox(option.label, option.choices, index=index, key=key)
            if option.choices
            else None
        )
    if isinstance(option, MultiChoice):
        return st.multiselect(option.label, option.choices, default=list(value), key=key)
    if isinstance(option, Number):
        return int(
            st.number_input(
                option.label,
                min_value=option.minimum,
                max_value=option.maximum,
                value=int(value),
                key=key,
            )
        )
    if isinstance(option, Toggle):
        return st.toggle(option.label, value=bool(value), key=key)
    raise TypeError(f"unsupported option {option!r}")


def render_panel(
    index: int, spec: PanelSpec, runs: Sequence[RunData], layout: Layout, editing: bool
) -> PanelSpec:
    """Draw one panel instance and return its spec with any option changes applied."""
    panel = PANELS.get(spec.panel)
    if panel is None:
        st.warning(f"Unknown panel '{spec.panel}' in this layout.")
        return spec
    options = dict(spec.options)
    with st.container(border=True):
        if panel.scope == "run":
            labels = [run.label for run in runs]
            saved = options.get(RUN_OPTION)
            chosen: str = saved if isinstance(saved, str) and saved in labels else labels[0]
            panel_runs = [runs[labels.index(chosen)]]
        else:
            panel_runs = list(runs)
        with st.expander(f"⚙ {panel.title}", expanded=False):
            if panel.description:
                st.caption(panel.description)
            if panel.scope == "run" and len(runs) > 1:
                chosen = st.selectbox("Run", labels, index=labels.index(chosen), key=f"{index}-run")
                options[RUN_OPTION] = chosen
                panel_runs = [runs[labels.index(chosen)]]
            resolved = panel.resolve_options(panel_runs, options)
            for option in panel.options(panel_runs):
                options[option.key] = option_widget(
                    option, resolved[option.key], f"{index}-{option.key}"
                )
            if editing:
                width = st.slider(
                    "Width (columns)",
                    1,
                    layout.columns,
                    min(spec.width, layout.columns),
                    key=f"{index}-w",
                )
                spec = replace(spec, width=width)
        problem = panel.missing(panel_runs)
        if problem:
            st.info(problem)
        else:
            output: object = panel.render(panel_runs, panel.resolve_options(panel_runs, options))
            if isinstance(output, Table):
                if output.caption:
                    st.caption(output.caption)
                st.dataframe(list(output.rows), width="stretch", hide_index=True)
            elif isinstance(output, Message):
                (st.warning if output.kind == "warning" else st.info)(output.text)
            else:
                st.plotly_chart(output, width="stretch", key=f"{index}-chart")
    return replace(
        spec, options={k: list(v) if isinstance(v, tuple) else v for k, v in options.items()}
    )


def edit_controls(index: int, count: int) -> str | None:
    up, down, remove = st.columns(3)
    if up.button("↑", key=f"{index}-up", disabled=index == 0, width="stretch"):
        return "up"
    if down.button("↓", key=f"{index}-down", disabled=index == count - 1, width="stretch"):
        return "down"
    if remove.button("✕ remove", key=f"{index}-remove", width="stretch"):
        return "remove"
    return None


def render_layout(layout: Layout, runs: Sequence[RunData], editing: bool) -> None:
    specs = list(layout.panels)
    action: tuple[int, str] | None = None
    row: list[tuple[int, PanelSpec]] = []
    rows: list[list[tuple[int, PanelSpec]]] = []
    for index, spec in enumerate(specs):
        width = min(spec.width, layout.columns)
        if sum(min(s.width, layout.columns) for _, s in row) + width > layout.columns:
            rows.append(row)
            row = []
        row.append((index, spec))
    if row:
        rows.append(row)
    for row in rows:
        columns = st.columns([min(spec.width, layout.columns) for _, spec in row])
        for column, (index, spec) in zip(columns, row, strict=True):
            with column:
                specs[index] = render_panel(index, spec, runs, layout, editing)
                if editing and (move := edit_controls(index, len(specs))):
                    action = (index, move)
    if action:
        index, move = action
        if move == "remove":
            specs.pop(index)
        else:
            other = index - 1 if move == "up" else index + 1
            specs[index], specs[other] = specs[other], specs[index]
    st.session_state.layout = replace(layout, panels=tuple(specs))
    if action:
        st.rerun()


def sidebar(layouts: dict[str, Layout]) -> tuple[Path, list[RunData], bool, int | None]:
    st.sidebar.title("AERIS")
    root = Path(st.sidebar.text_input("Runs directory", str(RUNS_ROOT)))
    try:
        summaries = list_runs(root)
    except (OSError, ValueError) as exc:
        st.sidebar.error(str(exc))
        return root, [], False, None
    labels = [Path(s.directory).name for s in summaries]
    selected = st.sidebar.multiselect("Runs", labels, default=labels[:1])
    runs = []
    for label in selected:
        try:
            runs.append(get_run(root / label))
        except (OSError, ValueError) as exc:
            st.sidebar.error(f"{label}: {exc}")

    st.sidebar.divider()
    names = list(layouts)
    current: Layout | None = st.session_state.get("layout")
    default = names.index(current.name) if current is not None and current.name in names else 0
    name = st.sidebar.selectbox("Dashboard", names, index=default)
    if current is None or st.session_state.get("layout_choice") != name:
        st.session_state.layout = layouts[name]
        st.session_state.layout_choice = name
    editing = st.sidebar.toggle("Edit layout")
    live = st.sidebar.toggle("Live refresh")
    interval = st.sidebar.number_input("Refresh every (s)", 2, 600, 10) if live else None
    return root, runs, editing, int(interval) if interval else None


def layout_editor(layouts: dict[str, Layout]) -> None:
    layout: Layout = st.session_state.layout
    with st.sidebar.expander("Layout", expanded=True):
        columns = st.number_input("Columns", 1, 4, layout.columns)
        choice = st.selectbox(
            "Add panel",
            list(PANELS),
            format_func=lambda n: f"{PANELS[n].title} ({PANELS[n].scope})",
        )
        if st.button("Add panel", width="stretch"):
            layout = replace(layout, panels=(*layout.panels, PanelSpec(panel=choice)))
        name = st.text_input("Save as", layout.name)
        layout = replace(layout, columns=int(columns))
        st.session_state.layout = layout
        if st.button("Save", type="primary", width="stretch"):
            path = save_layout(LAYOUT_DIR, replace(layout, name=name))
            st.session_state.layout_choice = name
            st.success(f"Saved {path}")


def main() -> None:
    st.set_page_config(page_title="AERIS dashboard", layout="wide")
    layouts = load_layouts(LAYOUT_DIR)
    root, runs, editing, interval = sidebar(layouts)
    if editing:
        layout_editor(layouts)
    dashboard_tab, runs_tab = st.tabs(["Dashboard", "Runs"])
    with runs_tab:
        try:
            summaries = list_runs(root)
            st.dataframe([s.to_dict() for s in summaries], width="stretch", hide_index=True)
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    with dashboard_tab:
        if not runs:
            st.info("Select one or more runs in the sidebar.")
            return

        @st.fragment(run_every=interval)
        def grid() -> None:
            # Re-read runs inside the fragment so live refresh picks up new records.
            fresh = [get_run(run.root) for run in runs]
            render_layout(st.session_state.layout, fresh, editing)

        grid()


main()
