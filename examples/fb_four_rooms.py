"""AERIS end-to-end reward-free FB diagnostic, NOT a paper reproduction.

Run: uv run --extra fb-torch python examples/fb_four_rooms.py --updates 250
Outputs in runs/fb_four_rooms/, or a user-selected --output directory.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from aeris.agents.fb import DiscreteFBAgent, DiscreteFBConfig, FBTransitionSampler
from aeris.data import JsonlRunRecorder
from aeris.domain.experiment import (ComponentSpec, EvaluationPlan, EvaluationTarget,
                                     ExperimentSpec, TrainingPlan)
from aeris.domain.results import RunManifest
from aeris.domain.task import TaskSpec
from aeris.environments import FourRooms
from aeris.experiments import collect_transitions, evaluate_goal_policy, train_updates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/fb_four_rooms"))
    p.add_argument("--dataset", choices=["oracle", "random"], default="oracle",
                   help="Explicit full-coverage one-step transitions or collected random experience")
    p.add_argument("--transitions", type=int, default=5000, help="Random collection size")
    p.add_argument("--updates", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--goal-row", type=int, default=9)
    p.add_argument("--goal-col", type=int, default=9)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {args.output}")
    if args.updates < 1 or args.batch_size < 1 or args.episodes < 1:
        raise ValueError("updates, batch-size and episodes must be positive")
    torch.set_num_threads(1)  # Fast and stable CPU behavior for a tiny diagnostic.
    env = FourRooms(max_episode_steps=args.max_steps)
    eval_env = FourRooms(max_episode_steps=args.max_steps)
    goal = (args.goal_row, args.goal_col)
    goal_observation = env.observation_for(goal)
    if args.dataset == "oracle":
        transitions = tuple(env.all_transitions())
    else:
        collected = collect_transitions(
            env, count=args.transitions, seed=args.seed,
            choose_action=lambda obs, rng: int(rng.integers(env.num_actions)),
        )
        transitions = collected.transitions
    # Fixed training data, no environment interaction after this point.
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_file = args.output / "dataset.npz"
    np.savez_compressed(
        dataset_file,
        observations=np.stack([t.observation for t in transitions]),
        actions=np.array([t.action for t in transitions], dtype=np.int64),
        next_observations=np.stack([t.next_observation for t in transitions]),
    )
    fingerprint = sha256(dataset_file.read_bytes()).hexdigest()

    task = TaskSpec(task_id=f"reach_{goal[0]}_{goal[1]}",
                    settings={"goal_row": goal[0], "goal_col": goal[1]})
    spec = ExperimentSpec(
        name="fb_four_rooms_diagnostic",
        agent=ComponentSpec(name="discrete_fb", settings={"representation_dim": 32}),
        environment=ComponentSpec(name="four_rooms", settings={"max_episode_steps": args.max_steps}),
        dataset=ComponentSpec(name="fixed_grid_transitions", settings={
            "origin": args.dataset, "count": len(transitions), "sha256": fingerprint,
        }),
        training=TrainingPlan(mode="offline", strategy=ComponentSpec(
            name="fixed_batch_updates", settings={"updates": args.updates, "batch_size": args.batch_size})),
        evaluation=EvaluationPlan(protocol=ComponentSpec(name="goal_reaching_v1", settings={
            "episodes": args.episodes, "max_steps": args.max_steps,
        }), targets=(EvaluationTarget(task=task),)),
        seed=args.seed,
    )
    run_id = "fb_four_rooms_seed_" + str(args.seed)
    recorder = JsonlRunRecorder(args.output)
    recorder.write_manifest(RunManifest(run_id=run_id, experiment=spec, status="running",
        provenance={"source": "AERIS demo, not Touati paper reproduction",
                    "numpy": np.__version__, "torch": torch.__version__,
                    "collection_seed": args.seed, "dataset_file": "dataset.npz",
                    "dataset_sha256": fingerprint}))
    try:
        agent = DiscreteFBAgent(DiscreteFBConfig(
            observation_dim=env.observation_dim, num_actions=env.num_actions,
            representation_dim=32, hidden_dim=128, seed=args.seed))
        provider = FBTransitionSampler(transitions, batch_size=args.batch_size, seed=args.seed + 2)
        match_goal = lambda obs: bool(np.array_equal(obs, goal_observation))
        initial = evaluate_goal_policy(
            agent, eval_env, condition=agent.goal_condition(goal_observation), is_goal=match_goal,
            episodes=args.episodes, max_steps=args.max_steps, run_id=run_id,
            task_id=task.task_id, checkpoint_id="initial_in_memory", seed=args.seed + 100)
        recorder.write_evaluation(initial)
        training = train_updates(agent, provider, updates=args.updates, run_id=run_id,
                                 recorder=recorder, log_every=max(1, args.updates // 10))
        # Trusted local-only torch checkpoint. Never torch.load an untrusted file.
        torch.save({"agent": agent.state_dict(), "sampler": provider.state_dict()},
                   args.output / "checkpoint.pt")
        final = evaluate_goal_policy(
            agent, eval_env, condition=agent.goal_condition(goal_observation), is_goal=match_goal,
            episodes=args.episodes, max_steps=args.max_steps, run_id=run_id,
            task_id=task.task_id, checkpoint_id="checkpoint.pt", seed=args.seed + 100)
        recorder.write_evaluation(final)
        recorder.write_manifest(RunManifest(run_id=run_id, experiment=spec,
            status="completed", selected_checkpoint="checkpoint.pt",
            provenance={"source": "AERIS demo, not Touati paper reproduction",
                        "numpy": np.__version__, "torch": torch.__version__,
                        "collection_seed": args.seed, "dataset_file": "dataset.npz",
                        "dataset_sha256": fingerprint}))
        print(f"Transitions: {len(transitions)} | gradient updates: {training.update_count}")
        for label, result in (("initial", initial), ("final", final)):
            print(label, {m.name: round(m.value, 4) for m in result.metrics})
        print(f"Saved run at {args.output}")
        print("Diagnostic only: no claim of reproducing published maze results.")
    except Exception as exc:
        recorder.write_manifest(RunManifest(run_id=run_id, experiment=spec, status="failed",
            error=f"{type(exc).__name__}: {exc}", provenance={"dataset_sha256": fingerprint}))
        raise
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
