"""Extract one-step FB representations for a probe set.

Runs inside third_party/onestep-fb's venv with that repo as the working directory, so it imports the repo's own
agent and dataset code. It must not import predictive_representations_rl.

Writes to --out:
    backward.npz        values [N, d]        B(s, a) at the probe's final state and action
    forward.npz         values [N, d]        F(s, a), mean over the forward ensemble (members in `ensemble`)
    latent.npz          values [tasks, d]    z inferred per task from reward-labelled validation data, as main.py does
    q_values.npz        values [N, tasks]    F(s, a) . z for each task
    extract_info.json   shapes, tasks, checkpoint
"""

import argparse
import importlib
import inspect
import json
import re
from pathlib import Path

import jax
import numpy as np

from agents import agents
from envs.exorl_utils import ALL_TASKS
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets, relabel_dataset
from utils.flax_utils import restore_agent


def load_agent(checkpoint, probe_observations, probe_actions):
    run_dir = checkpoint.parent
    flags = json.loads((run_dir / "flags.json").read_text())
    config = flags["agent"]
    epoch = int(re.findall(r"\d+", checkpoint.stem)[-1])

    example_batch = {"observations": probe_observations[:1], "actions": probe_actions[:1]}
    agent = agents[config["agent_name"]].create(flags["seed"], example_batch, config)
    return restore_agent(agent, str(run_dir), epoch), flags


# main.py clips dataset actions to [-1 + eps, 1 - eps] (make_env_and_datasets default); probes get the same.
ACTION_CLIP_EPS = inspect.signature(make_env_and_datasets).parameters["action_clip_eps"].default


def infer_latents(agent, flags):
    """Per-task z exactly as main.py computes it for zero-shot evaluation."""
    config = agent.config
    dataset_config = dict(config["dataset"])
    dataset_config["discount"] = config["discount"]
    dataset_class = getattr(importlib.import_module("utils.datasets"), dataset_config["dataset_class"])

    # The same call as main.py, so the validation data gets the same processing (its dataset_only path is broken).
    env_name = flags["env_name"]
    env, _, val_dataset = make_env_and_datasets(env_name, frame_stack=dataset_config["frame_stack"], add_info=True)
    domain = env_name.split("-")[-1]

    tasks, latents = [], []
    for task in ALL_TASKS[domain]:
        relabelled = relabel_dataset(f"{env_name}-{task}", env, val_dataset)
        dataset = dataset_class(Dataset.create(**relabelled), dataset_config)
        count = config["num_zero_shot_samples"]
        batch = dataset.sample(count, idxs=np.arange(count), relabeling=False, augmentation=False)
        tasks.append(task)
        latents.append(np.asarray(agent.infer_latent(batch)))

    return tasks, np.stack(latents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.probes) as probes:
        observations = probes["observations"][:, -1].astype(np.float32)
        actions = np.clip(probes["actions"][:, -1], -1 + ACTION_CLIP_EPS, 1 - ACTION_CLIP_EPS).astype(np.float32)

    agent, flags = load_agent(args.checkpoint, observations, actions)
    backward_fn = jax.jit(lambda s, a: agent.network.select("backward_repr")(s, actions=a))
    forward_fn = jax.jit(lambda s, a: agent.network.select("forward_repr")(s, actions=a))

    backward = np.asarray(backward_fn(observations, actions))
    forward_ensemble = np.asarray(forward_fn(observations, actions))  # [ensemble, N, d]
    forward = forward_ensemble.mean(axis=0)
    tasks, latents = infer_latents(agent, flags)
    q_values = np.einsum("nd,td->nt", forward, latents)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "backward.npz", values=backward)
    np.savez(args.out / "forward.npz", values=forward, ensemble=forward_ensemble)
    np.savez(args.out / "latent.npz", values=latents, tasks=np.array(tasks))
    np.savez(args.out / "q_values.npz", values=q_values, tasks=np.array(tasks))

    info = {
        "checkpoint": str(args.checkpoint),
        "tasks": tasks,
        "representations": {
            "backward": list(backward.shape),
            "forward": list(forward.shape),
            "latent": list(latents.shape),
            "q_values": list(q_values.shape),
        },
    }
    (args.out / "extract_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
