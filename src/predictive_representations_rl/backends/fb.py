from __future__ import annotations

import numpy as np

from predictive_representations_rl.core.data import RepresentationDataset


def representation_dataset_to_fb(dataset: RepresentationDataset) -> dict[str, np.ndarray]:

    if not dataset.trajectories:
        raise ValueError("RepresentationDataset contains no trajectories")

    observations = []
    next_observations = []
    actions = []
    terminals = []

    for trajectory in dataset.trajectories:
        if trajectory.num_steps == 0:
            raise ValueError("FB does not support empty trajectories")

        if not isinstance(trajectory.observations, np.ndarray):
            raise TypeError("FB supports only ndarray observations")

        # Canonical format:
        #
        # observations = [s_0, s_1, ..., s_T]
        # actions      = [a_0, a_1, ..., a_{T-1}]
        #
        # FB format:
        #
        # observation[t]      = s_t
        # next_observation[t] = s_{t+1}

        observations.append(trajectory.observations[:-1])
        next_observations.append(trajectory.observations[1:])
        actions.append(trajectory.actions)

        episode_ends = np.asarray(trajectory.episode_ends, dtype=np.float32).copy()

        # Even if this trajectory is a truncated piece of experience rather
        # than a true environment termination, FB must not sample goals across
        # the trajectory boundary.
        episode_ends[-1] = 1.0

        terminals.append(episode_ends)

    fb_dataset = {
        "observations": np.concatenate(observations, axis=0),
        "actions": np.concatenate(actions, axis=0),
        "next_observations": np.concatenate(next_observations, axis=0),
        "terminals": np.concatenate(terminals, axis=0)
    }

    expected_size = dataset.num_transitions

    for key, value in fb_dataset.items():
        if len(value) != expected_size:
            raise RuntimeError(f"FB field {key!r} has length {len(value)}, expected {expected_size}")

    return fb_dataset