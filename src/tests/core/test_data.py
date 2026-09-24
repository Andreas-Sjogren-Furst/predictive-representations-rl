import numpy as np

from predictive_representations_rl.core.data import (
    RepresentationDataset,
    RewardedTrajectory,
    TaskDataset,
    Trajectory,
)


def test_trajectory_structure():
    trajectory = Trajectory(
        observations=np.zeros((11, 4)),
        actions=np.zeros((10, 2)),
        terminated=np.zeros(10, dtype=bool),
        truncated=np.zeros(10, dtype=bool),
    )

    assert trajectory.num_steps == 10
    assert not trajectory.episode_ends.any()


def test_representation_dataset_counts_transitions():
    trajectory = Trajectory(
        observations=np.zeros((11, 4)),
        actions=np.zeros((10, 2)),
        terminated=np.zeros(10, dtype=bool),
        truncated=np.zeros(10, dtype=bool),
    )

    dataset = RepresentationDataset(
        trajectories=(trajectory, trajectory),
    )

    assert dataset.num_transitions == 20


def test_task_dataset():
    trajectory = Trajectory(
        observations=np.zeros((11, 4)),
        actions=np.zeros((10, 2)),
        terminated=np.zeros(10, dtype=bool),
        truncated=np.zeros(10, dtype=bool),
    )

    rewarded = RewardedTrajectory(
        trajectory=trajectory,
        rewards=np.ones(10),
    )

    dataset = TaskDataset(
        task_name="walk",
        trajectories=(rewarded,),
    )

    assert dataset.num_transitions == 10