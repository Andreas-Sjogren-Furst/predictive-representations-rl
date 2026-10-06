import h5py
import numpy as np
import pytest

from predictive_representations_rl.envs.dmc import point_mass_maze_factors
from predictive_representations_rl.probes.probe_set import ProbeSet, build_from_exorl, sample_windows


def test_windows_stay_inside_one_episode_and_skip_the_last_step():
    # Three episodes of 10 steps; the last step of each has a placeholder action.
    resets = np.zeros(30, bool)
    resets[[0, 10, 20]] = True

    starts = sample_windows(resets, 30, num=15, length=4, rng=np.random.default_rng(0))

    for start in starts:
        episode_end = (start // 10) * 10 + 9
        assert start + 4 - 1 < episode_end


def test_too_many_windows_is_an_error():
    resets = np.zeros(10, bool)
    resets[0] = True

    with pytest.raises(ValueError, match="windows"):
        sample_windows(resets, 10, num=100, length=4, rng=np.random.default_rng(0))


def test_build_from_exorl_and_round_trip(tmp_path):
    steps = 50
    observations = np.random.default_rng(0).uniform(-0.3, 0.3, (2 * steps, 4)).astype(np.float32)
    resets = np.zeros(2 * steps, bool)
    resets[[0, steps]] = True
    with h5py.File(tmp_path / "val.hdf5", "w") as file:
        file["observations"] = observations
        file["actions"] = np.arange(2 * steps, dtype=np.float32)[:, None].repeat(2, 1)
        file["resets"] = resets
        file["rewards/reach_top_left"] = np.arange(2 * steps, dtype=np.float32)

    probes = build_from_exorl(tmp_path / "val.hdf5", num=20, length=5, seed=0, factors=point_mass_maze_factors)

    assert probes.observations.shape == (20, 5, 4) and probes.actions.shape == (20, 5, 2)
    index = probes.metadata["source_index"]
    np.testing.assert_array_equal(probes.final_observations, observations[index])
    np.testing.assert_array_equal(probes.final_actions[:, 0], index)  # actions[t] belongs to observations[t]
    np.testing.assert_array_equal(probes.metadata["reward_reach_top_left"], index)
    np.testing.assert_array_equal(probes.metadata["x"], observations[index, 0])
    np.testing.assert_array_equal(probes.metadata["step"], index % steps)

    probes.save(tmp_path / "probes.npz")
    loaded = ProbeSet.load(tmp_path / "probes.npz")
    np.testing.assert_array_equal(loaded.observations, probes.observations)
    assert loaded.metadata.keys() == probes.metadata.keys()


def test_point_mass_maze_rooms_and_goal_distances():
    observations = np.array([[-0.2, 0.2, 0, 0], [0.2, 0.2, 0, 0], [-0.2, -0.2, 0, 0], [0.15, -0.15, 0.3, 0.4]])

    factors = point_mass_maze_factors(observations)

    np.testing.assert_array_equal(factors["room"], [0, 1, 2, 3])
    assert factors["dist_reach_bottom_right"][3] == 0
    assert factors["speed"][3] == pytest.approx(0.5)
