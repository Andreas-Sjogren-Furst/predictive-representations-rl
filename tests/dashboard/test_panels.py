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
from aeris.dashboard.panels.base import Message, MultiChoice, Stats, Table
from aeris.dashboard.panels.tables import SECTIONS, display
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
    assert isinstance(output, (Table, Stats, Message)) or output.data


def test_map_panels_check_optimality_and_animate(four_rooms_run):
    run = populate(four_rooms_run())
    policy = PANELS["policy_map"].render([run], PANELS["policy_map"].resolve_options([run], {}))
    assert "optimal choices 100% · optimal 100% · detour 0%" in policy.layout.title.text
    assert not any(trace.name == "suboptimal action" and trace.showlegend for trace in policy.data)
    animation = PANELS["trajectory"].render([run], PANELS["trajectory"].resolve_options([run], {}))
    assert len(animation.frames) == 3


def test_multi_run_panels_overlay_runs(four_rooms_run):
    runs = [populate(four_rooms_run("a")), populate(four_rooms_run("b"))]
    panel = PANELS["loss_curves"]
    curves = panel.render(
        runs, panel.resolve_options(runs, {"metrics": ("fb_loss",), "smoothing": 1})
    )
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
    layout = Layout(
        name="Run overview",
        columns=3,
        panels=(PanelSpec(panel="report", width=2),),
        shared={"task": "reach_1_1"},
    )
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
    "name",
    [
        "suite_scores",
        "suite_distribution",
        "task_policy",
        "goal_scores_map",
        "suite_tasks",
        "rollout",
        "fate_map",
    ],
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
        {"task": "room_top_left", "colour": "start score", "compare": True, "start": "9,9"},
        {"task": "random_dense_0", "colour": "visits"},
        {"actions": True},
        {"task": "lowest-scoring task", "show": "difference", "order": "suite order"},
        {"protocol": "all_states", "check_optimal": False},
    ):
        output = panel.render([data], panel.resolve_options([data], overrides))
        assert isinstance(output, Table) or output.data


def test_task_policy_marks_invisible_tasks(four_rooms_run):
    run = four_rooms_run()
    report = synthetic_suite_report()
    run.write_report(report)
    data = populate(run)
    first = report.tasks[0].task.task_id  # synthetic: no rewarded dataset evidence
    panel = PANELS["task_policy"]
    figure = panel.render(
        [data], panel.resolve_options([data], {"task": first, "protocol": "dataset_rewards"})
    )
    assert "no rewarded evidence" in figure.layout.title.text


def test_grouped_panels_aggregate_seeds(four_rooms_run):
    from aeris.dashboard.panels.grouping import group_choice

    runs = []
    for index in range(3):
        run = four_rooms_run(f"seed{index}")
        run.write_report(synthetic_suite_report(seed=index))
        runs.append(populate(run))
    assert group_choice(runs).choices == ("run", "name")  # nothing else differs between them
    curves = PANELS["loss_curves"].render(
        runs, {"metrics": ("fb_loss",), "smoothing": 1, "separate": True, "group_by": "name"}
    )
    assert [t.name for t in curves.data] == [
        "name=fb_demo (n=3) · fb_loss",
        "name=fb_demo (n=3) · fb_loss ± std",
    ]
    bars = PANELS["suite_scores"].render(runs, {"protocol": "dataset_rewards", "group_by": "name"})
    assert len(bars.data) == 1 and bars.data[0].error_y.visible
    table = PANELS["group_summary"].render(runs, {"group_by": "name"})
    assert "(n=3)" in table.rows[0]["name=fb_demo (n=3)"]


def test_behaviour_panels_roll_out_the_saved_suite_actions(four_rooms_run):
    from dataclasses import replace

    from aeris.domain import SuiteOutcome

    report = synthetic_suite_report()
    states = report.states
    goal = report.result("reach_1_1")
    oracle = build_goal_oracle(FourRooms().tabular_dynamics(), (1, 1), gamma=0.99)
    best = tuple(int(a) for a in oracle.optimal_policy())  # dynamics order = report order
    stuck = tuple(1 if s == (9, 9) else a for s, a in zip(states, best, strict=True))
    outcomes = {
        "dataset_rewards": replace(goal.outcomes["dataset_rewards"], actions=best),
        "all_states": replace(goal.outcomes["all_states"], actions=stuck),
    }
    assert all(isinstance(o, SuiteOutcome) for o in outcomes.values())
    tasks = tuple(replace(t, outcomes=outcomes) if t is goal else t for t in report.tasks)
    run = four_rooms_run()
    run.write_report(replace(report, tasks=tasks))
    data = populate(run)
    replay = PANELS["rollout"]
    figure = replay.render(
        [data], replay.resolve_options([data], {"task": "reach_1_1", "start": "9,9"})
    )
    assert "reaches the goal in 16 steps (shortest 16)" in figure.layout.title.text
    assert "0 suboptimal actions" in figure.layout.title.text and len(figure.frames) == 17
    figure = replay.render(
        [data],
        replay.resolve_options(
            [data],
            {"task": "reach_1_1", "start": "9,9", "protocol": "all_states", "compare": True},
        ),
    )
    title = figure.layout.title.text
    assert "stays at 9,9 from step 0" in title and "1 suboptimal action" in title
    assert "dataset_rewards: different path" in title
    fates = PANELS["fate_map"]
    figure = fates.render(
        [data], fates.resolve_options([data], {"task": "reach_1_1", "protocol": "all_states"})
    )
    assert "optimal 99% · detour 0% · late 0% · trapped 1%" in figure.layout.title.text
    policy = PANELS["task_policy"]
    figure = policy.render(
        [data], policy.resolve_options([data], {"task": "reach_1_1", "protocol": "all_states"})
    )
    marked = next(t for t in figure.data if t.name == "suboptimal action")
    rings = next(t for t in figure.data if t.name == "loop (where it ends up)")
    assert list(zip(marked.y, marked.x, strict=True)) == [(9, 9)]
    assert list(zip(rings.y, rings.x, strict=True)) == [(9, 9)]
    figure = replay.render(
        [data], replay.resolve_options([data], {"task": "reach_1_1", "protocol": "all_states"})
    )
    assert "from 9,9 (lowest-scoring start)" in figure.layout.title.text
    figure = fates.render(
        [data],
        fates.resolve_options(
            [data], {"task": "reach_1_1", "protocol": "all_states", "colour": "evidence difference"}
        ),
    )
    assert "differs from dataset_rewards at 1 of 68 starts" in figure.layout.title.text


def test_policy_map_marks_suboptimal_actions_and_trapped_starts(four_rooms_run):
    run = four_rooms_run()
    populate(run)
    dynamics = FourRooms().tabular_dynamics()
    oracle = build_goal_oracle(dynamics, (9, 9), gamma=0.99)
    actions = oracle.optimal_policy().copy()
    actions[dynamics.index_of((9, 8))] = 2  # left, away from the goal next door
    run.write_policy_snapshot("final", PolicySnapshot(states=oracle.states, actions=actions))
    data = load_run(run.root)
    panel = PANELS["policy_map"]
    figure = panel.render([data], panel.resolve_options([data], {}))
    title = figure.layout.title.text
    assert "optimal choices 99%" in title and "trapped 0%" not in title
    marked = next(t for t in figure.data if t.name == "suboptimal action")
    assert list(zip(marked.y, marked.x, strict=True)) == [(9, 8)]


def test_run_summary_shows_headlines_and_names_missing_analyses(four_rooms_run):
    run = four_rooms_run()
    data = populate(run)
    summary = PANELS["run_summary"].render([data], {})
    assert [stat.label for stat in summary.stats] == ["Status", "Setup"]
    assert "aeris analysis zero-shot" in summary.caption
    run.write_report(synthetic_suite_report())
    summary = PANELS["run_summary"].render([load_run(run.root)], {})
    labels = [stat.label for stat in summary.stats]
    assert {"Zero-shot score", "Zero-shot goal success", "Weakest family"} <= set(labels)
    assert "zero-shot" not in summary.caption


def test_settings_table_groups_sections_and_hides_provenance(four_rooms_run):
    data = populate(four_rooms_run())
    panel = PANELS["run_overview"]
    table = panel.render([data], panel.resolve_options([data], {}))
    sections = [row["section"] for row in table.rows]
    assert "provenance" not in sections and sections == sorted(sections, key=SECTIONS.index)
    everything = panel.render([data], {"sections": SECTIONS})
    assert {row["section"] for row in everything.rows} >= {"run", "agent"}


def test_report_numbers_are_rounded():
    assert [display(v) for v in (0.6716417910447762, 200.0, 0.0018377, 0.0, True)] == [
        "0.672",
        "200",
        "0.00184",
        "0",
        "True",
    ]


def test_loss_curves_draw_one_chart_per_metric(four_rooms_run):
    data = populate(four_rooms_run())
    panel = PANELS["loss_curves"]
    figure = panel.render(
        [data], panel.resolve_options([data], {"metrics": ["fb_loss", "fb_regularization"]})
    )
    assert [trace.yaxis for trace in figure.data] == ["y", "y2"]
    assert not any(trace.showlegend for trace in figure.data)  # one run: titles name the metrics
    overlaid = panel.render(
        [data],
        panel.resolve_options(
            [data], {"metrics": ["fb_loss", "fb_regularization"], "separate": False}
        ),
    )
    assert [trace.name for trace in overlaid.data] == ["fb_loss", "fb_regularization"]


def test_lowest_scoring_task_is_resolved_per_protocol(four_rooms_run):
    run = four_rooms_run()
    report = synthetic_suite_report()
    run.write_report(report)
    data = populate(run)
    for protocol in report.evidence_protocols:
        lowest = min(report.tasks, key=lambda t: t.outcomes[protocol].normalized_score)
        panel = PANELS["fate_map"]
        options = panel.resolve_options(
            [data], {"task": "lowest-scoring task", "protocol": protocol}
        )
        title = panel.render([data], options).layout.title.text
        assert f"{lowest.task.task_id} (lowest score) · {protocol}" in title


def test_suite_table_puts_the_lowest_score_first_with_the_protocol_gap(four_rooms_run):
    run = four_rooms_run()
    run.write_report(synthetic_suite_report())
    data = populate(run)
    panel = PANELS["suite_tasks"]
    table = panel.render([data], panel.resolve_options([data], {}))
    scores = [row["score · dataset_rewards"] for row in table.rows]
    assert scores == sorted(scores)
    row = table.rows[0]
    gap = row["gap · dataset_rewards − all_states"]
    assert gap == pytest.approx(
        row["score · dataset_rewards"] - row["score · all_states"], abs=2e-3
    )


def test_suite_scores_put_the_overall_mean_last(four_rooms_run):
    run = four_rooms_run()
    run.write_report(synthetic_suite_report())
    data = populate(run)
    panel = PANELS["suite_scores"]
    figure = panel.render([data], panel.resolve_options([data], {}))
    assert figure.data[0].x[-1] == "all tasks" and "all" not in figure.data[0].x


def test_reward_maps_diverge_around_zero_only_for_signed_rewards():
    from aeris.dashboard.panels.grid import REWARD_DIVERGING, REWARD_SCALE, reward_colours

    signed = reward_colours(np.array([-2.0, 0.5, 1.0]))
    assert (signed["zmin"], signed["zmax"]) == (-2.0, 2.0)
    assert signed["colorscale"] == REWARD_DIVERGING
    distance = reward_colours(np.array([-1.0, -0.5, 0.0]))
    assert (distance["zmin"], distance["zmax"]) == (-1.0, 0.0)
    assert distance["colorscale"] == REWARD_SCALE


def test_fate_map_marks_optimal_loops(four_rooms_run):
    from dataclasses import replace

    from aeris.analysis.task_scores import optimal_solution

    env = FourRooms()
    dynamics = env.tabular_dynamics()
    report = synthetic_suite_report()
    room = report.result("room_top_left")
    rewards = np.empty(dynamics.num_states)
    rewards[[dynamics.index_of(p) for p in env.positions]] = env.state_rewards(room.task)
    best = optimal_solution(dynamics, rewards, gamma=report.gamma).optimal_actions.argmax(axis=1)
    order = [dynamics.index_of(state) for state in report.states]
    actions = tuple(int(best[i]) for i in order)
    outcomes = {p: replace(o, actions=actions) for p, o in room.outcomes.items()}
    tasks = tuple(replace(t, outcomes=outcomes) if t is room else t for t in report.tasks)
    run = four_rooms_run()
    run.write_report(replace(report, tasks=tasks))
    data = populate(run)
    panel = PANELS["fate_map"]
    figure = panel.render([data], panel.resolve_options([data], {"task": "room_top_left"}))
    assert "100% of starts reach an optimal loop" in figure.layout.title.text
    ticks = next(t for t in figure.data if t.type == "heatmap" and t.colorbar.ticktext)
    assert all("✓ optimal" in tick for tick in ticks.colorbar.ticktext if tick.startswith("loop"))
