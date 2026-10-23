"""AERIS end-to-end reward-free FB diagnostic, NOT a paper reproduction.

Run: uv run --extra fb-torch python examples/fb_four_rooms.py --updates 250
Outputs in runs/fb_four_rooms/, or a user-selected --output directory.
Inspect the run afterwards with examples/four_rooms_report.py.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from aeris.agents.fb import DiscreteFBAgent, DiscreteFBConfig, FBTransitionSampler
from aeris.analysis.four_rooms import save_policy_snapshot
from aeris.data import JsonlRunRecorder
from aeris.domain import (
    ComponentSpec,
    EvaluationPlan,
    EvaluationRecord,
    EvaluationTarget,
    ExperimentSpec,
    RunManifest,
    TaskSpec,
    TrainingPlan,
    Transition,
)
from aeris.environments import FourRooms
from aeris.experiments import collect_transitions, evaluate_goal_policy, train_updates

REPRESENTATION_DIM = 32
HIDDEN_DIM = 128


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("runs/fb_four_rooms"))
    p.add_argument(
        "--dataset", choices=["oracle", "random"], default="oracle",
        help="Explicit full-coverage one-step transitions or collected random experience",
    )
    p.add_argument("--transitions", type=int, default=5000, help="Random collection size")
    p.add_argument("--updates", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--trajectories", type=int, default=3,
        help="Number of evaluation episodes to save per checkpoint (0 disables recording)",
    )
    p.add_argument(
        "--plots", action="store_true",
        help="Render report PNGs after completing the experiment (requires matplotlib)",
    )
    p.add_argument("--goal-row", type=int, default=9)
    p.add_argument("--goal-col", type=int, default=9)
    args = p.parse_args()
    if args.trajectories < 0:
        p.error("--trajectories must be nonnegative")
    if args.updates < 1 or args.batch_size < 1 or args.episodes < 1:
        p.error("--updates, --batch-size and --episodes must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        p.error(f"output directory must be empty: {args.output}")
    return args


def build_dataset(env: FourRooms, args: argparse.Namespace) -> tuple[Transition, ...]:
    if args.dataset == "oracle":
        return tuple(env.all_transitions())
    return collect_transitions(
        env, count=args.transitions, seed=args.seed,
        choose_action=lambda obs, rng: int(rng.integers(env.num_actions)),
    ).transitions


def save_dataset(path: Path, transitions: Sequence[Transition]) -> str:
    """Write the fixed training data and return its SHA-256 fingerprint."""
    np.savez_compressed(
        path,
        observations=np.stack([t.observation for t in transitions]),
        actions=np.array([t.action for t in transitions], dtype=np.int64),
        next_observations=np.stack([t.next_observation for t in transitions]),
    )
    return sha256(path.read_bytes()).hexdigest()


def build_spec(args: argparse.Namespace, task: TaskSpec, *, count: int, fingerprint: str) -> ExperimentSpec:
    return ExperimentSpec(
        name="fb_four_rooms_diagnostic",
        agent=ComponentSpec(name="discrete_fb", settings={"representation_dim": REPRESENTATION_DIM}),
        environment=ComponentSpec(name="four_rooms", settings={"max_episode_steps": args.max_steps}),
        dataset=ComponentSpec(
            name="fixed_grid_transitions",
            settings={"origin": args.dataset, "count": count, "sha256": fingerprint},
        ),
        training=TrainingPlan(
            mode="offline",
            strategy=ComponentSpec(
                name="fixed_batch_updates",
                settings={"updates": args.updates, "batch_size": args.batch_size},
            ),
        ),
        evaluation=EvaluationPlan(
            protocol=ComponentSpec(
                name="goal_reaching_v1",
                settings={"episodes": args.episodes, "max_steps": args.max_steps},
            ),
            targets=(EvaluationTarget(task=task),),
        ),
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)  # Fast and stable CPU behavior for a tiny diagnostic.
    env = FourRooms(max_episode_steps=args.max_steps)
    eval_env = FourRooms(max_episode_steps=args.max_steps)
    goal = (args.goal_row, args.goal_col)
    goal_observation = env.observation_for(goal)

    # Fixed training data, no environment interaction after this point.
    transitions = build_dataset(env, args)
    args.output.mkdir(parents=True, exist_ok=True)
    fingerprint = save_dataset(args.output / "dataset.npz", transitions)

    task = TaskSpec(task_id=f"reach_{goal[0]}_{goal[1]}", settings={"goal_row": goal[0], "goal_col": goal[1]})
    spec = build_spec(args, task, count=len(transitions), fingerprint=fingerprint)
    run_id = f"fb_four_rooms_seed_{args.seed}"
    provenance = {
        "source": "AERIS demo, not Touati paper reproduction",
        "numpy": np.__version__,
        "torch": torch.__version__,
        "collection_seed": args.seed,
        "dataset_file": "dataset.npz",
        "dataset_sha256": fingerprint,
    }
    recorder = JsonlRunRecorder(args.output)
    recorder.write_manifest(RunManifest(run_id=run_id, experiment=spec, status="running", provenance=provenance))

    def evaluate(agent: DiscreteFBAgent, checkpoint_id: str, snapshot: str) -> EvaluationRecord:
        condition = agent.goal_condition(goal_observation)
        save_policy_snapshot(agent, eval_env, condition, args.output / snapshot)
        record = evaluate_goal_policy(
            agent, eval_env, condition=condition,
            is_goal=lambda obs: bool(np.array_equal(obs, goal_observation)),
            episodes=args.episodes, max_steps=args.max_steps, run_id=run_id,
            task_id=task.task_id, checkpoint_id=checkpoint_id, seed=args.seed + 100,
            on_trajectory=recorder.write_trajectory if args.trajectories else None,
            max_trajectories=args.trajectories,
        )
        recorder.write_evaluation(record)
        return record

    try:
        agent = DiscreteFBAgent(
            DiscreteFBConfig(
                observation_dim=env.observation_dim, num_actions=env.num_actions,
                representation_dim=REPRESENTATION_DIM, hidden_dim=HIDDEN_DIM, seed=args.seed,
            )
        )
        sampler = FBTransitionSampler(transitions, batch_size=args.batch_size, seed=args.seed + 2)
        initial = evaluate(agent, "initial_in_memory", "policy_initial.npz")
        training = train_updates(
            agent, sampler, updates=args.updates, run_id=run_id,
            recorder=recorder, log_every=max(1, args.updates // 10),
        )
        # Trusted local-only torch checkpoint. Never torch.load an untrusted file.
        torch.save({"agent": agent.state_dict(), "sampler": sampler.state_dict()}, args.output / "checkpoint.pt")
        final = evaluate(agent, "checkpoint.pt", "policy_final.npz")
        artifacts = {
            "policy_initial": "policy_initial.npz",
            "policy_final": "policy_final.npz",
            "trajectories": "trajectories.jsonl" if args.trajectories else None,
            "metric_file": "training_metrics.jsonl",
            "evaluation_file": "evaluations.jsonl",
        }
        recorder.write_manifest(
            RunManifest(
                run_id=run_id, experiment=spec, status="completed", selected_checkpoint="checkpoint.pt",
                provenance={**provenance, "diagnostic_artifacts": artifacts},
            )
        )
    except Exception as exc:
        recorder.write_manifest(
            RunManifest(
                run_id=run_id, experiment=spec, status="failed",
                error=f"{type(exc).__name__}: {exc}", provenance=provenance,
            )
        )
        raise
    finally:
        env.close()
        eval_env.close()

    print(f"Transitions: {len(transitions)} | gradient updates: {training.update_count}")
    for label, result in (("initial", initial), ("final", final)):
        print(label, {m.name: round(m.value, 4) for m in result.metrics})
    print(f"Saved run at {args.output}")
    print("Diagnostic only: no claim of reproducing published maze results.")
    # Plotting is deliberately outside the training exception/manifest logic.
    # A rendering failure must not mark successfully completed training as failed.
    if args.plots:
        from aeris.visualization.four_rooms import render_four_rooms_report

        made = render_four_rooms_report(args.output)
        print("Rendered:", ", ".join(str(f) for f in made))


if __name__ == "__main__":
    main()
