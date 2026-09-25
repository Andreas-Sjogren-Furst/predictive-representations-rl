import numpy as np

from predictive_representations_rl.backends.fb import (
    representation_dataset_to_fb,
)
from predictive_representations_rl.core.data import (
    RepresentationDataset,
    Trajectory,
)


def test_representation_dataset_to_fb():
    trajectory_1 = Trajectory(
        observations=np.array(
            [
                [0.0],
                [1.0],
                [2.0],
            ]
        ),
        actions=np.array(
            [
                [10.0],
                [11.0],
            ]
        ),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
    )

    trajectory_2 = Trajectory(
        observations=np.array(
            [
                [100.0],
                [101.0],
                [102.0],
                [103.0],
            ]
        ),
        actions=np.array(
            [
                [20.0],
                [21.0],
                [22.0],
            ]
        ),
        terminated=np.array([False, False, False]),
        truncated=np.array([False, False, False]),
    )

    dataset = RepresentationDataset(
        trajectories=(
            trajectory_1,
            trajectory_2,
        )
    )

    fb_dataset = representation_dataset_to_fb(dataset)

    np.testing.assert_array_equal(
        fb_dataset["observations"],
        np.array(
            [
                [0.0],
                [1.0],
                [100.0],
                [101.0],
                [102.0],
            ]
        ),
    )

    np.testing.assert_array_equal(
        fb_dataset["next_observations"],
        np.array(
            [
                [1.0],
                [2.0],
                [101.0],
                [102.0],
                [103.0],
            ]
        ),
    )

    np.testing.assert_array_equal(
        fb_dataset["actions"],
        np.array(
            [
                [10.0],
                [11.0],
                [20.0],
                [21.0],
                [22.0],
            ]
        ),
    )

    np.testing.assert_array_equal(
        fb_dataset["terminals"],
        np.array(
            [
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    )