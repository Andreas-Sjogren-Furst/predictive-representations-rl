"""Workflows for completed single-goal Four Rooms runs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aeris.agents.fb.reports import FBCheckpointReport
from aeris.analysis.goal_policy import (
    TabularPolicyAnalysis,
    analyze_policy_snapshots,
    build_goal_oracle,
)
from aeris.data.artifacts import TransitionFile
from aeris.data.run_directory import RunDirectory
from aeris.domain.reports import ZeroShotOutcome, ZeroShotReport, ZeroShotTaskResult
from aeris.domain.results import RunManifest
from aeris.domain.task import TaskEvidenceRef, TaskSpec
from aeris.domain.validation import require_int
from aeris.environments.four_rooms import FourRooms, Position
from aeris.experiments.zero_shot import evaluate_zero_shot
from aeris.workflows._runtime import use_single_cpu_thread

# Run conventions shared by training (four_rooms_training.py) and the analyses here:
# each evaluation label names a policy snapshot and the checkpoint id it was evaluated as.
EVALUATION_CHECKPOINTS = {"initial": "initial_in_memory", "final": "checkpoint.pt"}
SNAPSHOT_LABELS = tuple(EVALUATION_CHECKPOINTS)

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

    use_single_cpu_thread()
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
    _require_recorded_config(run.manifest, asdict(agent.config))
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


ZERO_SHOT_EVIDENCE = {
    "goal_observation": (
        "one sample, the goal observation with reward 1; privileged (it names the goal state) "
        "and, for FB, equal to conditioning on B(goal)"
    ),
    "dataset_rewards": (
        "the run's reward-free pretraining observations relabelled with the task reward "
        "r(s) = 1 if s is the goal else 0; the zero-shot setting"
    ),
}


def evaluate_zero_shot_goals(
    run_ref: RunRef,
    *,
    goals: Sequence[Position] | None = None,
    samples: int | None = None,
    episodes: int = 10,
    seed: int = 0,
) -> ZeroShotReport:
    """Infer each goal task from evidence, then evaluate the frozen policy (needs torch).

    For every goal (default: every traversable cell) and every protocol in
    ``ZERO_SHOT_EVIDENCE`` the agent infers its condition through the
    TaskConditioner contract and is evaluated on the same seeded start states.
    ``samples`` draws a fixed random subset of the pretraining observations as
    the relabelled evidence (default: all of them). Inference never sees
    evaluation rollouts, and the agent is never updated. Saves
    ``analysis/zero_shot_goals_v1.json``.
    """
    from aeris.agents.fb.checkpoint import load_agent

    use_single_cpu_thread()

    run = open_four_rooms_run(run_ref)
    dataset_spec = run.manifest.experiment.dataset
    if dataset_spec is None:
        raise ValueError("zero-shot evidence needs the run's saved pretraining dataset")
    fingerprint = dataset_spec.settings.get("sha256")
    dataset = TransitionFile(run.store.dataset_path, expected_sha256=fingerprint)
    agent = load_agent(run.store.checkpoint_path)
    agent.config.spec.require_compatible(run.env.spec, consumer="FB checkpoint")
    _require_recorded_config(run.manifest, asdict(agent.config))

    observations = dataset.observations
    if samples is not None:
        require_int("samples", samples, minimum=1)
        if samples > len(observations):
            raise ValueError(f"samples must be at most the dataset size ({len(observations)})")
        chosen = np.random.default_rng(seed).choice(len(observations), size=samples, replace=False)
        observations = observations[np.sort(chosen)]
    sample_states = [run.env.position_of(obs) for obs in observations]
    eval_env = FourRooms(max_episode_steps=run.max_steps)
    evidence_source = f"{run.store.dataset_path.name} sha256:{fingerprint}"
    checkpoint = run.store.checkpoint_path.name

    tasks = []
    for goal in goals if goals is not None else run.env.positions:
        task = run.env.goal_task(goal)
        goal_obs = run.env.observation_for(goal)
        rewards = [1.0 if state == goal else 0.0 for state in sample_states]
        evidence = {
            "goal_observation": [(goal_obs, 1.0)],
            "dataset_rewards": list(zip(observations, rewards, strict=True)),
        }
        outcomes = {}
        for name, pairs in evidence.items():
            record = evaluate_zero_shot(
                agent,
                eval_env,
                task=task,
                evidence=pairs,
                evidence_ref=TaskEvidenceRef(source_id=evidence_source, split=name),
                is_goal=_reaches(goal_obs),
                episodes=episodes,
                max_steps=run.max_steps,
                run_id=run.manifest.run_id,
                checkpoint_id=checkpoint,
                seed=seed,
            )
            metrics = {m.name: m.value for m in record.metrics}
            outcomes[name] = ZeroShotOutcome(
                evidence_samples=len(pairs),
                rewarded_samples=sum(1 for _, r in pairs if r != 0),
                success_rate=metrics["success_rate"],
                mean_episode_length=metrics["mean_episode_length"],
            )
        tasks.append(ZeroShotTaskResult(task_id=task.task_id, goal=goal, outcomes=outcomes))

    report = ZeroShotReport(
        run_id=run.manifest.run_id,
        checkpoint=checkpoint,
        evidence_source=evidence_source,
        evidence_protocols=ZERO_SHOT_EVIDENCE,
        episodes=episodes,
        max_steps=run.max_steps,
        seed=seed,
        tasks=tuple(tasks),
        mean_success_rate={
            name: float(np.mean([t.outcomes[name].success_rate for t in tasks]))
            for name in ZERO_SHOT_EVIDENCE
        },
    )
    run.store.write_report(report)
    return report


def _reaches(goal_observation: np.ndarray) -> Callable[[Any], bool]:
    def is_goal(observation: Any) -> bool:
        return bool(np.array_equal(observation, goal_observation))

    return is_goal


def _require_recorded_config(manifest: RunManifest, config: dict[str, Any]) -> None:
    """Refuse a checkpoint that contradicts the agent settings recorded in the manifest."""
    recorded = manifest.experiment.agent.settings
    mismatched = [
        f"{key} (manifest {recorded[key]!r}, checkpoint {config.get(key)!r})"
        for key in recorded
        if key != "device" and recorded[key] != config.get(key)
    ]
    if mismatched:
        raise ValueError(
            "checkpoint does not match the run's agent settings: " + "; ".join(mismatched)
        )


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
    for label, checkpoint in EVALUATION_CHECKPOINTS.items():
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


def render_zero_shot_figures(run_ref: RunRef, report: ZeroShotReport) -> tuple[Path, ...]:
    """Plot per-goal zero-shot success maps into ``figures/``."""
    from aeris.visualization.zero_shot import render_zero_shot_success

    path = _store(run_ref).figures_dir / "zero_shot_success.png"
    return (render_zero_shot_success(report, path),)
