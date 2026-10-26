"""Every registered panel renders from typed run data, without Streamlit."""

import json

import numpy as np
import pytest

from conftest import synthetic_representations, synthetic_suite_report

pytest.importorskip("plotly")
from aeris.analysis.goal_policy import build_goal_oracle
from aeris.dashboard.data import load_run, run_stamp
from aeris.dashboard.layouts import BUILT_IN, Layout, PanelSpec, load_layouts, save_layout
from aeris.dashboard.panels import PANELS
from aeris.dashboard.panels.base import Message, MultiChoice, Table
from aeris.dashboard.panels.training import smooth
from aeris.domain import (
    EvaluationRecord,
    EvaluationTrajectory,
    MetricValue,
    PolicySnapshot,
    TrainingMetricRecord,
    TrajectoryStep,
)
from aeris.environments import FourRooms


def add_representations(run, seed=0):
    """A synthetic but well-formed representation artifact over every Four Rooms cell."""
    states = FourRooms().positions
    reps = synthetic_representations(states=states, goals=((9, 9), (1, 1)), d=8, a=4, seed=seed)
    run.write_arrays(reps.ARTIFACT_NAME, *reps.to_arrays())


def populate(run):
    env = FourRooms()
    for step in (10, 20, 30):
        run.write_training_metrics(
            TrainingMetricRecord(
                "run-42",
                "gradient_updates",
                step,
                (
                    MetricValue("fb_loss", -step, "u", "i"),
                    MetricValue("fb_regularization", 0.1, "u", "i"),
                ),
            )
        )
    run.write_evaluation(
        EvaluationRecord(
            "run-42",
            "checkpoint.pt",
            "reach_9_9",
            "goal",
            0,
            2,
            (MetricValue("success_rate", 0.5, "fraction", "mean"),),
        )
    )
    oracle = build_goal_oracle(env.tabular_dynamics(), (9, 9), gamma=0.99)
    run.write_policy_snapshot(
        "final", PolicySnapshot(states=oracle.states, actions=oracle.optimal_policy())
    )
    path = [(8, 8), (8, 9), (9, 9)]
    run.write_trajectory(
        EvaluationTrajectory(
            run_id="run-42",
            task_id="reach_9_9",
            checkpoint_id="checkpoint.pt",
            protocol_id="goal",
            seed=0,
            episode_index=0,
            initial_observation=env.observation_for(path[0]),
            steps=tuple(
                TrajectoryStep(
                    env.observation_for(a),
                    3,
                    env.observation_for(b),
                    None,
                    False,
                    False,
                    b == (9, 9),
                )
                for a, b in zip(path, path[1:], strict=False)
            ),
            success=True,
        )
    )
    return load_run(run.root)


@pytest.mark.parametrize("name", sorted(PANELS))
def test_every_panel_renders_or_explains_what_is_missing(name, four_rooms_run):
    run = populate(four_rooms_run())
    panel = PANELS[name]
    problem = panel.missing([run])
    if problem:
        assert "aeris analysis" in problem  # missing reports name the command to create them
        return
    output = panel.render([run], panel.resolve_options([run], {}))
    assert isinstance(output, (Table, Message)) or output.data


def test_map_panels_check_optimality_and_animate(four_rooms_run):
    run = populate(four_rooms_run())
    policy = PANELS["policy_map"].render([run], PANELS["policy_map"].resolve_options([run], {}))
    assert "optimal choices 100%" in policy.layout.title.text
    animation = PANELS["trajectory"].render([run], PANELS["trajectory"].resolve_options([run], {}))
    assert len(animation.frames) == 3


def test_multi_run_panels_overlay_runs(four_rooms_run):
    runs = [populate(four_rooms_run("a")), populate(four_rooms_run("b"))]
    curves = PANELS["loss_curves"].render(runs, {"metrics": ("fb_loss",), "smoothing": 1})
    assert [trace.name for trace in curves.data] == ["a · fb_loss", "b · fb_loss"]
    table = PANELS["run_comparison"].render(runs, {})
    assert [row["run"] for row in table.rows] == ["a", "b"]


def test_saved_options_fall_back_when_no_longer_valid(four_rooms_run):
    run = populate(four_rooms_run())
    panel = PANELS["loss_curves"]
    options = panel.resolve_options([run], {"metrics": ["gone"], "smoothing": 3})
    metric_option = next(o for o in panel.options([run]) if isinstance(o, MultiChoice))
    assert options["metrics"] == metric_option.default and options["smoothing"] == 3


def test_smoothing_is_a_trailing_mean():
    assert smooth(np.array([1.0, 3.0, 5.0]), 2).tolist() == [1.0, 2.0, 4.0]


def test_run_stamp_changes_when_the_run_changes(four_rooms_run):
    run = four_rooms_run()
    before = run_stamp(run.root)
    run.write_evaluation(
        EvaluationRecord(
            "run-42", "c", "t", "p", 0, 1, (MetricValue("success_rate", 1.0, "f", "m"),)
        )
    )
    assert run_stamp(run.root) != before


def test_layouts_roundtrip_and_override_built_ins(tmp_path):
    layout = Layout(name="Run overview", columns=3, panels=(PanelSpec(panel="report", width=2),))
    path = save_layout(tmp_path, layout)
    assert path.name == "run_overview.json"
    assert json.loads(path.read_text())["columns"] == 3
    layouts = load_layouts(tmp_path)
    assert layouts["Run overview"] == layout
    assert layouts["Compare runs"] == next(b for b in BUILT_IN if b.name == "Compare runs")
    assert all(spec.panel in PANELS for built_in in BUILT_IN for spec in built_in.panels)


@pytest.mark.parametrize(
    "name", ["projection", "similarity", "b_spectrum", "occupancy_maps", "value_map"]
)
def test_representation_panels_render_from_the_artifact(name, four_rooms_run):
    run = four_rooms_run()
    assert (
        PANELS[name].missing([populate(run)]) is not None
    )  # names `aeris analysis representations`
    add_representations(run)
    data = populate(run)
    panel = PANELS[name]
    assert panel.missing([data]) is None
    for overrides in (
        {},
        {"dimensions": 3, "color": "distance to goal"},
        {"action": "→", "log": True},
    ):
        figure = panel.render([data], panel.resolve_options([data], overrides))
        assert figure.data


def test_every_representation_kind_projects(four_rooms_run):
    from aeris.dashboard.panels.representations import REPRESENTATIONS

    run = four_rooms_run()
    add_representations(run)
    data = populate(run)
    for kind in REPRESENTATIONS:
        figure = PANELS["projection"].render(
            [data], PANELS["projection"].resolve_options([data], {"representation": kind})
        )
        assert figure.layout.title.text.startswith(kind.split(":")[0])


@pytest.mark.parametrize(
    "name", ["suite_scores", "suite_distribution", "task_policy", "goal_scores_map", "suite_tasks"]
)
def test_suite_panels_render_from_the_report(name, four_rooms_run):
    run = four_rooms_run()
    panel = PANELS[name]
    assert "aeris analysis zero-shot" in panel.missing([populate(run)])
    run.write_report(synthetic_suite_report())
    data = populate(run)
    assert panel.missing([data]) is None
    for overrides in (
        {},
        {"task": "reach_1_1", "metric": "goal success"},
        {"family": "room"},
    ):
        output = panel.render([data], panel.resolve_options([data], overrides))
        assert isinstance(output, Table) or output.data


def test_task_policy_marks_invisible_tasks(four_rooms_run):
    run = four_rooms_run()
    report = synthetic_suite_report()
    run.write_report(report)
    data = populate(run)
    first = report.tasks[0].task.task_id  # synthetic: no rewarded dataset evidence
    figure = PANELS["task_policy"].render([data], {"task": first, "protocol": "dataset_rewards"})
    assert "no rewarded evidence" in figure.layout.title.text
