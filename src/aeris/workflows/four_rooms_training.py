"""Train discrete FB offline on Four Rooms and save a self-describing run directory.

An AERIS diagnostic, NOT a reproduction of the FB paper's maze experiments.
Needs the ``fb-torch`` extra.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np

from aeris.agents.fb.batch import FBTransitionSampler
from aeris.agents.fb.config import DiscreteFBConfig
from aeris.data.artifacts import write_transitions_npz
from aeris.data.run_directory import EVALUATIONS, TRAINING_METRICS, TRAJECTORIES, RunDirectory
from aeris.domain.experience import Transition
from aeris.domain.experiment import (
    ComponentSpec,
    EvaluationPlan,
    EvaluationTarget,
    ExperimentSpec,
    TrainingPlan,
)
from aeris.domain.results import EvaluationRecord, RunManifest
from aeris.domain.task import TaskSpec
from aeris.domain.validation import require_int
from aeris.environments.four_rooms import FourRooms, Position
from aeris.experiments import (
    collect_transitions,
    evaluate_goal_policy,
    record_policy_snapshot,
    train_updates,
)
from aeris.workflows._runtime import use_single_cpu_thread
from aeris.workflows.four_rooms import EVALUATION_CHECKPOINTS
from aeris.workflows.provenance import code_revision

DatasetOrigin = Literal["oracle", "random"]


def run_name(name: str, seed: int) -> str:
    """Run id and default directory name of one seed of an experiment."""
    return f"{name}_seed{seed}"


@dataclass(frozen=True, slots=True, kw_only=True)
class FourRoomsFBTraining:
    """Everything that defines one training run; recorded in its manifest."""

    output: Path
    name: str = "fb_four_rooms"  # experiment (group) name shared by a configuration's seeds
    dataset: DatasetOrigin = "oracle"  # full state-action coverage, or random collection
    transitions: int = 5000  # collection size for the random dataset
    updates: int = 250
    batch_size: int = 128
    episodes: int = 20  # evaluation episodes per checkpoint
    max_steps: int = 60
    seed: int = 7
    trajectories: int = 3  # evaluation episodes saved per checkpoint (0 disables)
    goal: Position = (9, 9)
    representation_dim: int = 100  # upstream discrete-maze settings
    hidden_dim: int = 256
    log_every: int | None = None  # updates between logged metrics; default: 100 points per run

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name):
            raise ValueError("name must be letters, digits, '_', '.' or '-' (it names directories)")
        if self.dataset not in get_args(DatasetOrigin):
            raise ValueError(f"dataset must be one of {get_args(DatasetOrigin)}")
        for name in ("transitions", "updates", "batch_size", "episodes", "max_steps"):
            require_int(name, getattr(self, name), minimum=1)
        for name in ("seed", "trajectories"):
            require_int(name, getattr(self, name))
        if self.log_every is not None:
            require_int("log_every", self.log_every, minimum=1)
        object.__setattr__(self, "output", Path(self.output))

    @property
    def run_id(self) -> str:
        return run_name(self.name, self.seed)


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    run: RunDirectory
    transition_count: int
    update_count: int
    initial: EvaluationRecord
    final: EvaluationRecord


def train_four_rooms_fb(settings: FourRoomsFBTraining) -> TrainingOutcome:
    """Collect fixed data, train FB offline, evaluate before and after, save everything.

    The manifest records the full agent configuration, the dataset fingerprint,
    library versions, the AERIS git commit (and whether the checkout had
    uncommitted changes) and the status; a failure after the run starts is
    recorded as ``failed``.
    """
    import torch

    from aeris.agents.fb.agent import DiscreteFBAgent
    from aeris.agents.fb.checkpoint import save_checkpoint

    if settings.output.exists() and any(settings.output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {settings.output}")
    use_single_cpu_thread()
    env = FourRooms(max_episode_steps=settings.max_steps)
    eval_env = FourRooms(max_episode_steps=settings.max_steps)
    task = env.goal_task(settings.goal)
    goal_observation = env.observation_for(settings.goal)
    config = DiscreteFBConfig(
        observation_dim=env.observation_dim,
        num_actions=env.num_actions,
        representation_dim=settings.representation_dim,
        hidden_dim=settings.hidden_dim,
        seed=settings.seed,
    )

    # Fixed training data, no environment interaction after this point.
    transitions = _dataset(env, settings)
    run = RunDirectory(settings.output)
    fingerprint = write_transitions_npz(run.dataset_path, transitions)
    spec = _experiment_spec(settings, task, config, count=len(transitions), fingerprint=fingerprint)
    provenance: dict[str, Any] = {
        "source": "AERIS demo, not Touati paper reproduction",
        "numpy": np.__version__,
        "torch": torch.__version__,
        **code_revision(),
        "collection_seed": settings.seed,
        "dataset_file": run.dataset_path.name,
        "dataset_sha256": fingerprint,
    }
    run.write_manifest(
        RunManifest(
            run_id=settings.run_id, experiment=spec, status="running", provenance=provenance
        )
    )

    def evaluate(agent: DiscreteFBAgent, label: str) -> EvaluationRecord:
        condition = agent.goal_condition(goal_observation)
        snapshot = record_policy_snapshot(
            agent, eval_env.positions, eval_env.observation_for, condition=condition
        )
        snapshot.require_action_space(eval_env.spec.action)
        run.write_policy_snapshot(label, snapshot)
        record = evaluate_goal_policy(
            agent,
            eval_env,
            condition=condition,
            is_goal=lambda obs: bool(np.array_equal(obs, goal_observation)),
            episodes=settings.episodes,
            max_steps=settings.max_steps,
            run_id=settings.run_id,
            task_id=task.task_id,
            checkpoint_id=EVALUATION_CHECKPOINTS[label],
            seed=settings.seed + 100,
            on_trajectory=run.write_trajectory if settings.trajectories else None,
            max_trajectories=settings.trajectories,
        )
        run.write_evaluation(record)
        return record

    try:
        agent = DiscreteFBAgent(config)
        sampler = FBTransitionSampler(
            transitions, batch_size=settings.batch_size, seed=settings.seed + 2
        )
        initial = evaluate(agent, "initial")
        training = train_updates(
            agent,
            sampler,
            updates=settings.updates,
            run_id=settings.run_id,
            recorder=run,
            log_every=settings.log_every or max(1, settings.updates // 100),
        )
        save_checkpoint(run.checkpoint_path, agent, sampler=sampler)
        final = evaluate(agent, "final")
        artifacts = {
            "policy_initial": run.policy_snapshot_path("initial").name,
            "policy_final": run.policy_snapshot_path("final").name,
            "trajectories": TRAJECTORIES if settings.trajectories else None,
            "metric_file": TRAINING_METRICS,
            "evaluation_file": EVALUATIONS,
        }
        run.write_manifest(
            RunManifest(
                run_id=settings.run_id,
                experiment=spec,
                status="completed",
                selected_checkpoint=run.checkpoint_path.name,
                provenance={**provenance, "diagnostic_artifacts": artifacts},
            )
        )
    except Exception as exc:
        run.write_manifest(
            RunManifest(
                run_id=settings.run_id,
                experiment=spec,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                provenance=provenance,
            )
        )
        raise
    finally:
        env.close()
        eval_env.close()
    return TrainingOutcome(run, len(transitions), training.update_count, initial, final)


def _dataset(env: FourRooms, settings: FourRoomsFBTraining) -> tuple[Transition, ...]:
    if settings.dataset == "oracle":
        return tuple(env.all_transitions())
    return collect_transitions(
        env,
        count=settings.transitions,
        seed=settings.seed,
        choose_action=lambda obs, rng: int(rng.integers(env.num_actions)),
    ).transitions


def _experiment_spec(
    settings: FourRoomsFBTraining,
    task: TaskSpec,
    config: DiscreteFBConfig,
    *,
    count: int,
    fingerprint: str,
) -> ExperimentSpec:
    return ExperimentSpec(
        name=settings.name,
        # The full configuration, so the manifest alone says how the agent was built.
        agent=ComponentSpec(name="discrete_fb", settings=asdict(config)),
        environment=ComponentSpec(
            name="four_rooms", settings={"max_episode_steps": settings.max_steps}
        ),
        dataset=ComponentSpec(
            name="fixed_grid_transitions",
            settings={"origin": settings.dataset, "count": count, "sha256": fingerprint},
        ),
        training=TrainingPlan(
            mode="offline",
            strategy=ComponentSpec(
                name="fixed_batch_updates",
                settings={"updates": settings.updates, "batch_size": settings.batch_size},
            ),
        ),
        evaluation=EvaluationPlan(
            protocol=ComponentSpec(
                name="goal_reaching_v1",
                settings={"episodes": settings.episodes, "max_steps": settings.max_steps},
            ),
            targets=(EvaluationTarget(task=task),),
        ),
        seed=settings.seed,
    )
