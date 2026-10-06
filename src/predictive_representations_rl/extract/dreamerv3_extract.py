"""Extract DreamerV3 world-model representations for a probe set.

Runs inside the DreamerV3 venv with third_party/Offline_vs_Online_in_MBRL as the working directory, so it imports
the repo's own agent code. It must not import predictive_representations_rl.

The RSSM posterior is filtered over each probe window, as during training: at step t the model sees the previous
latent state, the previous action a_{t-1}, and the observation o_t. The first step starts from the model's learned
initial state with a zero action (is_first). The representation is the posterior at the final step.

Writes to --out (and the same files with an `_untrained` suffix, from the model at initialisation):
    deter.npz           values [N, deter]           h, the GRU state
    stoch.npz           values [N, stoch * classes] z, the one-hot categorical sample, flattened
    model_state.npz     values [N, deter + stoch * classes]  concat(h, z), the actor/critic input
    extract_info.json   shapes, observation keys, checkpoint
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path.cwd()
sys.path[:0] = [str(REPO / "dreamerv3"), str(REPO)]

import embodied  # noqa: E402
from dreamerv3 import agent as agt  # noqa: E402
from dreamerv3 import train as dreamer_train  # noqa: E402

SPECIAL_KEYS = ("reward", "is_first", "is_last", "is_terminal")


def load_agent(logdir, checkpoint):
    config = embodied.Config.load(logdir / "config.yaml")
    if os.environ.get("JAX_PLATFORMS") == "cpu":
        config = config.update({"jax.platform": "cpu"})
    if config.encoder.cnn_keys != "$^":
        raise NotImplementedError("pixel-based Dreamer runs need rendered probe images; only state encoders are supported")

    env = dreamer_train.make_env(config)
    obs_space, act_space = env.obs_space, env.act_space
    env.close()

    agent = agt.Agent(obs_space, act_space, embodied.Counter(), config)
    return agent, obs_space, config


def restore(agent, checkpoint):
    saved = embodied.Checkpoint()
    saved.agent = agent
    saved.load(checkpoint, keys=["agent"])


def split_observations(flat, obs_space):
    """Split flat probe observations [N, T, D] into the env's observation dict, in the env's key order."""
    keys = [k for k, space in obs_space.items() if k not in SPECIAL_KEYS and not k.startswith("log_") and len(space.shape) == 1]
    sizes = [obs_space[k].shape[0] for k in keys]
    if sum(sizes) != flat.shape[-1]:
        raise ValueError(f"probe observations have {flat.shape[-1]} dims, env keys {dict(zip(keys, sizes))} have {sum(sizes)}")

    parts = np.split(flat, np.cumsum(sizes)[:-1], axis=-1)
    return dict(zip(keys, parts)), keys


def filter_posterior(agent, observations, actions, obs_space):
    """Run the posterior over windows [N, T]; return the final latent state."""
    num, length = observations.shape[:2]
    obs_dict, _ = split_observations(observations.astype(np.float32), obs_space)

    # The world model's learned initial latent and a zero action, as obs_step uses when is_first is set.
    state, prev_action = agent.get_init_state(num)
    for t in range(length):
        prior = agent.get_prior(state, prev_action)
        obs = {key: value[:, t] for key, value in obs_dict.items()}
        obs.update(
            reward=np.zeros(num, np.float32),
            is_first=np.full(num, t == 0),
            is_last=np.zeros(num, bool),
            is_terminal=np.zeros(num, bool),
        )
        state, _ = agent.encode_and_get_post(obs, prior)  # (posterior, prior)
        prev_action = actions[:, t].astype(np.float32)

    return {key: np.asarray(value) for key, value in state.items()}


def representations(agent, observations, actions, obs_space):
    state = filter_posterior(agent, observations, actions, obs_space)
    deter = state["deter"]
    stoch = state["stoch"].reshape(len(deter), -1)
    return {
        "deter": {"values": deter},
        "stoch": {"values": stoch, "logit": state["logit"]},
        "model_state": {"values": np.concatenate([deter, stoch], axis=-1)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    agent, obs_space, config = load_agent(args.checkpoint.parent, args.checkpoint)
    with np.load(args.probes) as probes:
        observations, actions = probes["observations"], probes["actions"]

    # The untrained control first (same architecture, config and seed, at initialisation), then the checkpoint.
    outputs = {f"{name}_untrained": arrays for name, arrays in representations(agent, observations, actions, obs_space).items()}
    restore(agent, args.checkpoint)
    outputs.update(representations(agent, observations, actions, obs_space))

    args.out.mkdir(parents=True, exist_ok=True)
    for name, arrays in outputs.items():
        np.savez(args.out / f"{name}.npz", **arrays)

    info = {
        "checkpoint": str(args.checkpoint),
        "task": config.task,
        "observation_keys": split_observations(observations[:1, :1], obs_space)[1],
        "window_length": int(observations.shape[1]),
        "representations": {name: list(arrays["values"].shape) for name, arrays in outputs.items()},
    }
    (args.out / "extract_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
