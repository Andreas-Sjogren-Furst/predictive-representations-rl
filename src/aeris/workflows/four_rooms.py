"""Workflows for completed single-goal Four Rooms runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aeris.agents.fb.reports import FBCheckpointReport
from aeris.analysis.goal_policy import (
    TabularPolicyAnalysis,
    analyze_policy_snapshots,
    build_goal_oracle,
)
from aeris.data.artifacts import TransitionFile
from aeris.data.run_directory import RunDirectory
from aeris.domain.results import RunManifest
from aeris.domain.task import TaskSpec
from aeris.environments.four_rooms import FourRooms, Position

# Run conventions written by examples/fb_four_rooms.py.
SNAPSHOT_LABELS = ("initial", "final")
TRAJECTORY_CHECKPOINTS = {"initial": "initial_in_memory", "final": "checkpoint.pt"}

RunRef = str | Path | RunDirectory


def _store(run_ref: RunRef) -> RunDirectory:
    return run_ref if isinstance(run_ref, RunDirectory) else RunDirectory(run_ref)


@dataclass(frozen=True, slots=True)
class FourRoomsRun:
    """A run that passed the eligibility checks, with the values workflows need."""

    store: RunDirectory
    manifest: RunManifest
    env: FourRooms
    task: TaskSpec
    goal: Position
    max_steps: int


def open_four_rooms_run(run_ref: RunRef) -> FourRoomsRun:
    """Open a run, requiring a completed Four Rooms experiment with exactly one goal task."""
    store = _store(run_ref)
    manifest = store.read_manifest()
    if manifest.status != "completed":
        raise ValueError("run must be completed")
    experiment = manifest.experiment
    if experiment.environment is None or experiment.environment.name != "four_rooms":
        raise ValueError("only Four Rooms runs are supported")
    if experiment.evaluation is None or len(experiment.evaluation.targets) != 1:
        raise ValueError("exactly one evaluation target is required")
    if "max_steps" not in experiment.evaluation.protocol.settings:
        raise ValueError("the evaluation protocol does not record max_steps")
    env = FourRooms()
    task = experiment.evaluation.targets[0].task
    return FourRoomsRun(
        store=store,
        manifest=manifest,
        env=env,
        task=task,
        goal=env.goal_from_task(task),
        max_steps=int(experiment.evaluation.protocol.settings["max_steps"]),
    )


def analyze_saved_policies(run_ref: RunRef, *, gamma: float = 0.99) -> TabularPolicyAnalysis:
    """Compare the run's saved policy snapshots with the exact optimal policy, without torch.

    Saves ``analysis/tabular_policy_v1.json``; the original artifacts are unchanged.
    """
    run = open_four_rooms_run(run_ref)
    snapshots = {}
    for label in SNAPSHOT_LABELS:
        snapshot = run.store.read_policy_snapshot(label)
        if snapshot is not None:
            snapshot.require_action_space(run.env.spec.action)
            snapshots[label] = snapshot
    if not snapshots:
        raise ValueError("no saved policy snapshots were found")
    analysis = analyze_policy_snapshots(
        build_goal_oracle(run.env.tabular_dynamics(), run.goal, gamma=gamma),
        snapshots,
        max_steps=run.max_steps,
        run_id=run.manifest.run_id,
        task_id=run.task.task_id,
        snapshot_names={label: run.store.policy_snapshot_path(label).name for label in snapshots},
    )
    run.store.write_report(analysis.report)
    return analysis


def diagnose_fb_checkpoint(run_ref: RunRef) -> FBCheckpointReport:
    """Validate the run's trusted FB checkpoint against exact occupancies (needs torch).

    Requires the saved full-coverage oracle dataset, verified by fingerprint and
    by content rather than by its label. Saves ``analysis/fb_checkpoint_v1.json``.
    """
    from aeris.agents.fb.checkpoint import load_agent
    from aeris.analysis.fb_representation import diagnose_fb_representation

    run = open_four_rooms_run(run_ref)
    dataset = run.manifest.experiment.dataset
    settings = dataset.settings if dataset is not None else {}
    if settings.get("origin") != "oracle":
        raise ValueError("occupancy comparison requires the saved uniform-coverage oracle dataset")
    transitions = TransitionFile(run.store.dataset_path, expected_sha256=settings.get("sha256"))
    run.env.require_full_coverage(transitions.iter_transitions())
    agent = load_agent(run.store.checkpoint_path)
    agent.config.spec.require_compatible(run.env.spec, consumer="FB checkpoint")
    final_snapshot = run.store.read_policy_snapshot("final")
    if final_snapshot is not None:
        final_snapshot.require_action_space(run.env.spec.action)
    report = diagnose_fb_representation(
        agent,
        run.env,
        build_goal_oracle(run.env.tabular_dynamics(), run.goal, gamma=agent.config.gamma),
        max_steps=run.max_steps,
        run_id=run.manifest.run_id,
        checkpoint=run.store.checkpoint_path.name,
        final_snapshot=final_snapshot,
    )
    run.store.write_report(report)
    return report


def render_run_figures(run_ref: RunRef) -> tuple[Path, ...]:
    """Plot every recorded artifact of a run into ``figures/``; no torch or live policy."""
    from aeris.visualization.four_rooms import (
        render_evaluation_success,
        render_policy_map,
        render_training_metrics,
        render_trajectory,
    )

    run = open_four_rooms_run(run_ref)
    store, output = run.store, run.store.figures_dir
    made: list[Path] = []
    training = list(store.read_training_metrics())
    if training:
        made.append(render_training_metrics(training, output / "training.png"))
    evaluations = list(store.read_evaluations())
    if evaluations:
        made.append(render_evaluation_success(evaluations, output / "success_rate.png"))
    for label in SNAPSHOT_LABELS:
        snapshot = store.read_policy_snapshot(label)
        if snapshot is not None:
            name = store.policy_snapshot_path(label).stem
            made.append(render_policy_map(snapshot, run.goal, output / f"{name}.png", name=name))
    trajectories = list(store.read_trajectories())
    for label, checkpoint in TRAJECTORY_CHECKPOINTS.items():
        first = next((t for t in trajectories if t.checkpoint_id == checkpoint), None)
        if first is not None:
            made.append(render_trajectory(first, run.goal, output / f"trajectory_{label}.png"))
    return tuple(made)


def render_policy_figures(run_ref: RunRef, analysis: TabularPolicyAnalysis) -> tuple[Path, ...]:
    """Plot the optimal policy and each snapshot's policy gap into ``figures/``."""
    from aeris.visualization.policy_diagnostics import render_tabular_policy_figures

    return render_tabular_policy_figures(analysis, _store(run_ref).figures_dir)


def render_fb_figures(run_ref: RunRef, report: FBCheckpointReport) -> tuple[Path, ...]:
    """Plot the FB action-alignment map into ``figures/``."""
    from aeris.visualization.fb_checkpoint import render_fb_action_alignment

    path = _store(run_ref).figures_dir / "fb_action_alignment.png"
    return (render_fb_action_alignment(report, path),)
