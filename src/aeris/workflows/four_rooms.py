"""Workflows for completed single-goal Four Rooms runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aeris.agents.fb.reports import FBCheckpointReport
from aeris.agents.fb.representations import FBRepresentations
from aeris.analysis.goal_policy import (
    TabularPolicyAnalysis,
    analyze_policy_snapshots,
    build_goal_oracle,
)
from aeris.data.artifacts import TransitionFile
from aeris.data.run_directory import RunDirectory
from aeris.domain.reports import SuiteOutcome, SuiteTaskResult, ZeroShotSuiteReport
from aeris.domain.results import RunManifest
from aeris.domain.task import TaskSpec, task_family
from aeris.domain.validation import require_int
from aeris.environments.four_rooms import FourRooms, Position
from aeris.experiments.snapshots import record_policy_snapshot
from aeris.workflows._runtime import use_single_cpu_thread

if TYPE_CHECKING:
    from aeris.agents.fb.agent import DiscreteFBAgent

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


def _training_data(run: FourRoomsRun, purpose: str) -> TransitionFile:
    """The run's saved training dataset, verified against the manifest's fingerprint."""
    spec = run.manifest.experiment.dataset
    if spec is None:
        raise ValueError(f"{purpose} needs the run's saved training dataset")
    return TransitionFile(run.store.dataset_path, expected_sha256=spec.settings.get("sha256"))


def _load_checkpoint(run: FourRoomsRun) -> DiscreteFBAgent:
    """The run's trusted FB checkpoint, checked against the environment and the manifest."""
    from aeris.agents.fb.checkpoint import load_agent

    use_single_cpu_thread()
    agent = load_agent(run.store.checkpoint_path)
    agent.config.spec.require_compatible(run.env.spec, consumer="FB checkpoint")
    _require_recorded_config(run.manifest, asdict(agent.config))
    return agent


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
    from aeris.analysis.fb_representation import diagnose_fb_representation

    run = open_four_rooms_run(run_ref)
    dataset = run.manifest.experiment.dataset
    if dataset is None or dataset.settings.get("origin") != "oracle":
        raise ValueError("occupancy comparison requires the saved uniform-coverage oracle dataset")
    run.env.require_full_coverage(_training_data(run, "occupancy comparison").iter_transitions())
    agent = _load_checkpoint(run)
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


SUITE_EVIDENCE = {
    "dataset_rewards": (
        "the run's reward-free pretraining observations relabelled with the task reward r(s); "
        "the zero-shot setting"
    ),
    "all_states": (
        "every state once with its reward; privileged: the full reward function, with no "
        "dependence on what the training data covered"
    ),
}


def evaluate_task_suite(
    run_ref: RunRef,
    *,
    suite: str = "four_rooms_v1",
    samples: int | None = None,
    seed: int = 0,
) -> ZeroShotSuiteReport:
    """Infer every task of a suite from evidence and score the frozen policy exactly (needs torch).

    For each task and each protocol in ``SUITE_EVIDENCE``, the agent infers a
    condition through the TaskConditioner contract and acts greedily through
    the Agent contract (so any zero-shot agent can be scored). Its policy is
    then evaluated exactly on the task's reward against the optimal and the
    random policy. ``samples`` draws a fixed random subset of the pretraining
    observations as the relabelled evidence (default: all). Saves
    ``analysis/zero_shot_suite_v1.json``.
    """
    from aeris.analysis.task_scores import score_policy, task_baseline
    from aeris.environments.four_rooms_tasks import task_suite

    tasks = task_suite(suite)
    run = open_four_rooms_run(run_ref)
    dataset = _training_data(run, "zero-shot evidence")
    agent = _load_checkpoint(run)
    dynamics = run.env.tabular_dynamics()
    observations = dataset.observations
    if samples is not None:
        require_int("samples", samples, minimum=1)
        if samples > len(observations):
            raise ValueError(f"samples must be at most the dataset size ({len(observations)})")
        chosen = np.random.default_rng(seed).choice(len(observations), size=samples, replace=False)
        observations = observations[np.sort(chosen)]
    sample_states = [dynamics.index_of(run.env.position_of(obs)) for obs in observations]
    state_observations = [run.env.observation_for(state) for state in dynamics.states]
    gamma = agent.config.gamma

    results = []
    for task in tasks.tasks:
        rewards = run.env.state_rewards(task)
        family = task_family(task)
        goal_index = dynamics.index_of(run.env.goal_from_task(task)) if family == "goal" else None
        evidence = {
            "dataset_rewards": [
                (obs, float(rewards[i])) for obs, i in zip(observations, sample_states, strict=True)
            ],
            "all_states": [
                (obs, float(r)) for obs, r in zip(state_observations, rewards, strict=True)
            ],
        }
        baseline = task_baseline(dynamics, rewards, gamma=gamma)
        outcomes = {}
        for protocol, pairs in evidence.items():
            condition = agent.infer_condition(task, pairs)
            snapshot = record_policy_snapshot(
                agent, dynamics.states, run.env.observation_for, condition=condition
            )
            actions = np.asarray(snapshot.actions_for(dynamics.states))
            score = score_policy(
                dynamics,
                actions,
                rewards,
                baseline,
                gamma=gamma,
                goal_index=goal_index,
                max_steps=run.max_steps,
            )
            outcomes[protocol] = SuiteOutcome(
                normalized_score=score.normalized_score,
                policy_value=score.policy_value,
                goal_success=score.goal_success,
                evidence_samples=len(pairs),
                rewarded_samples=sum(1 for _, r in pairs if r != 0),
                actions=tuple(int(a) for a in actions),
            )
        results.append(
            SuiteTaskResult(
                task=task,
                family=family,
                optimal_value=baseline.optimal_value,
                random_value=baseline.random_value,
                outcomes=outcomes,
            )
        )

    report = ZeroShotSuiteReport(
        run_id=run.manifest.run_id,
        checkpoint=run.store.checkpoint_path.name,
        suite=tasks.name,
        suite_description=tasks.description,
        gamma=gamma,
        max_steps=run.max_steps,
        evidence_source=f"{run.store.dataset_path.name} sha256:{dataset.sha256}",
        evidence_protocols=SUITE_EVIDENCE,
        states=dynamics.states,
        tasks=tuple(results),
        mean_score=_mean_scores(results),
    )
    run.store.write_report(report)
    return report


def _mean_scores(results: Sequence[SuiteTaskResult]) -> dict[str, dict[str, float | None]]:
    """Mean normalized score per family (and over all tasks) and evidence protocol."""
    groups: dict[str, list[SuiteTaskResult]] = {"all": list(results)}
    for result in results:
        groups.setdefault(result.family, []).append(result)
    means: dict[str, dict[str, float | None]] = {}
    for family, members in groups.items():
        means[family] = {}
        for protocol in SUITE_EVIDENCE:
            scores = [r.outcomes[protocol].normalized_score for r in members]
            defined = [score for score in scores if score is not None]
            means[family][protocol] = float(np.mean(defined)) if defined else None
    return means


def export_fb_representations(
    run_ref: RunRef, *, goals: Sequence[Position] | None = None
) -> FBRepresentations:
    """Save B for every state and F, policy and exact density per goal (needs torch).

    The state density rho is the frequency of each cell among the run's training
    observations, so the learned-vs-exact comparison is valid for any dataset,
    not only the uniform oracle one. Saves ``analysis/fb_representations_v1.npz``,
    which the dashboard's representation panels read.
    """
    from aeris.analysis.fb_representation import compute_fb_representations

    run = open_four_rooms_run(run_ref)
    dataset = _training_data(run, "representations (for the state density)")
    agent = _load_checkpoint(run)
    dynamics = run.env.tabular_dynamics()
    counts = np.zeros(dynamics.num_states)
    for observation in dataset.observations:
        counts[dynamics.index_of(run.env.position_of(observation))] += 1
    representations = compute_fb_representations(
        agent,
        run.env,
        dynamics,
        goals if goals is not None else dynamics.states,
        state_density=counts / counts.sum(),
        run_id=run.manifest.run_id,
        checkpoint=run.store.checkpoint_path.name,
    )
    run.store.write_arrays(FBRepresentations.ARTIFACT_NAME, *representations.to_arrays())
    return representations


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
