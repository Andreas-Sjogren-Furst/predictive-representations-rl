"""AERIS end-to-end reward-free FB diagnostic, NOT a paper reproduction.

Run: uv run --extra fb-torch python examples/fb_four_rooms.py --updates 250
Outputs in runs/fb_four_rooms/, or a user-selected --output directory.
Inspect the run afterwards with examples/four_rooms_report.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from aeris.agents.fb import DiscreteFBAgent, DiscreteFBConfig, FBTransitionSampler
from aeris.agents.fb.checkpoint import save_checkpoint
from aeris.data import RunDirectory, write_transitions_npz
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
from aeris.experiments import (
    collect_transitions,
    evaluate_goal_policy,
    record_policy_snapshot,
    train_updates,
)

REPRESENTATION_DIM = 32
HIDDEN_DIM = 128


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output", type=Path, default=Path("runs/fb_four_rooms"))
    p.add_argument(
        "--dataset",
        choices=["oracle", "random"],
        default="oracle",
        help="Explicit full-coverage one-step transitions or collected random experience",
    )
    p.add_argument("--transitions", type=int, default=5000, help="Random collection size")
    p.add_argument("--updates", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--trajectories",
        type=int,
        default=3,
        help="Number of evaluation episodes to save per checkpoint (0 disables recording)",
    )
    p.add_argument(
        "--plots",
        action="store_true",
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
        env,
        count=args.transitions,
        seed=args.seed,
        choose_action=lambda obs, rng: int(rng.integers(env.num_actions)),
    ).transitions


def build_spec(
    args: argparse.Namespace, task: TaskSpec, *, count: int, fingerprint: str
) -> ExperimentSpec:
    return ExperimentSpec(
        name="fb_four_rooms_diagnostic",
        agent=ComponentSpec(
            name="discrete_fb", settings={"representation_dim": REPRESENTATION_DIM}
        ),
        environment=ComponentSpec(
            name="four_rooms", settings={"max_episode_steps": args.max_steps}
        ),
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
    run = RunDirectory(args.output)
    fingerprint = write_transitions_npz(run.dataset_path, transitions)

    task = env.goal_task(goal)
    spec = build_spec(args, task, count=len(transitions), fingerprint=fingerprint)
    run_id = f"fb_four_rooms_seed_{args.seed}"
    provenance = {
        "source": "AERIS demo, not Touati paper reproduction",
        "numpy": np.__version__,
        "torch": torch.__version__,
        "collection_seed": args.seed,
        "dataset_file": run.dataset_path.name,
        "dataset_sha256": fingerprint,
    }
    run.write_manifest(
        RunManifest(run_id=run_id, experiment=spec, status="running", provenance=provenance)
    )

    def evaluate(agent: DiscreteFBAgent, checkpoint_id: str, phase: str) -> EvaluationRecord:
        condition = agent.goal_condition(goal_observation)
        snapshot = record_policy_snapshot(
            agent, eval_env.positions, eval_env.observation_for, condition=condition
        )
        snapshot.require_action_space(eval_env.spec.action)
        run.write_policy_snapshot(phase, snapshot)
        record = evaluate_goal_policy(
            agent,
            eval_env,
            condition=condition,
            is_goal=lambda obs: bool(np.array_equal(obs, goal_observation)),
            episodes=args.episodes,
            max_steps=args.max_steps,
            run_id=run_id,
            task_id=task.task_id,
            checkpoint_id=checkpoint_id,
            seed=args.seed + 100,
            on_trajectory=run.write_trajectory if args.trajectories else None,
            max_trajectories=args.trajectories,
        )
        run.write_evaluation(record)
        return record

    try:
        agent = DiscreteFBAgent(
            DiscreteFBConfig(
                observation_dim=env.observation_dim,
                num_actions=env.num_actions,
                representation_dim=REPRESENTATION_DIM,
                hidden_dim=HIDDEN_DIM,
                seed=args.seed,
            )
        )
        sampler = FBTransitionSampler(transitions, batch_size=args.batch_size, seed=args.seed + 2)
        initial = evaluate(agent, "initial_in_memory", "initial")
        training = train_updates(
            agent,
            sampler,
            updates=args.updates,
            run_id=run_id,
            recorder=run,
            log_every=max(1, args.updates // 10),
        )
        checkpoint = save_checkpoint(run.checkpoint_path, agent, sampler=sampler)
        final = evaluate(agent, checkpoint.name, "final")
        artifacts = {
            "policy_initial": "policy_initial.npz",
            "policy_final": "policy_final.npz",
            "trajectories": "trajectories.jsonl" if args.trajectories else None,
            "metric_file": "training_metrics.jsonl",
            "evaluation_file": "evaluations.jsonl",
        }
        run.write_manifest(
            RunManifest(
                run_id=run_id,
                experiment=spec,
                status="completed",
                selected_checkpoint="checkpoint.pt",
                provenance={**provenance, "diagnostic_artifacts": artifacts},
            )
        )
    except Exception as exc:
        run.write_manifest(
            RunManifest(
                run_id=run_id,
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

    print(f"Transitions: {len(transitions)} | gradient updates: {training.update_count}")
    for label, result in (("initial", initial), ("final", final)):
        print(label, {m.name: round(m.value, 4) for m in result.metrics})
    print(f"Saved run at {args.output}")
    print("Diagnostic only: no claim of reproducing published maze results.")
    # Plotting is deliberately outside the training exception/manifest logic.
    # A rendering failure must not mark successfully completed training as failed.
    if args.plots:
        from aeris.workflows.four_rooms import render_run_figures

        made = render_run_figures(args.output)
        print("Rendered:", ", ".join(str(f) for f in made))


if __name__ == "__main__":
    main()
