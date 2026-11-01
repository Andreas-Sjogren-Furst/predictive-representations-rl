import numpy as np
import pytest

from aeris.analysis import network_analysis as na
from aeris.analysis.state_labels import four_rooms_labels
from aeris.environments import FourRooms
from conftest import synthetic_network_features


def test_ranks_count_the_directions_used():
    rng = np.random.default_rng(0)
    three = rng.normal(size=(50, 3)) @ rng.normal(size=(3, 20))
    singular = na.singular_values(three)
    assert na.srank(singular) == 3 and 1 < na.effective_rank(singular) <= 3
    assert na.effective_rank(na.singular_values(np.ones((5, 4)))) == 0.0
    scores, share = na.pca(three, 2)
    assert scores.shape == (50, 2) and share.sum() <= 1 and share[0] >= share[1]
    assert na.dead_units(np.c_[three, np.zeros(50)]) == pytest.approx(1 / 21)


def test_probes_are_scored_on_held_out_states():
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(68, 5))
    # More units than states: a training fit is always perfect, a held-out one is not.
    x = latent @ rng.normal(size=(5, 100)) + 0.05 * rng.normal(size=(68, 100))
    linear = latent[:, :3] @ np.array([1.0, -2.0, 0.5])
    assert na.probe(x, linear).score > 0.9
    assert na.probe(x, rng.normal(size=68)).score < 0.2  # noise is not decodable
    ring = np.sin(3 * latent[:, 0])
    flat = latent[:, :2]
    assert na.probe(flat, ring, kind="kernel").score > na.probe(flat, ring).score
    classes = (latent[:, 0] > 0).astype(int)
    result = na.probe(x, classes, categorical=True)
    assert result.metric == "accuracy" and result.score > 0.85 and result.chance >= 0.5
    anything = np.ones((68, 2), dtype=bool)  # every class accepted: always right
    assert na.probe(x, classes, categorical=True, accepted=anything).score == 1.0
    assert na.probe(x, classes, kind="knn", categorical=True).predictions.shape == (68,)
    with pytest.raises(ValueError, match="unknown probe"):
        na.probe(x, linear, kind="tree")


def test_clusters_recover_separated_groups():
    rng = np.random.default_rng(2)
    truth = np.repeat(np.arange(3), 10)
    x = rng.normal(scale=0.1, size=(30, 4)) + np.eye(4)[truth] * 5
    for labels in (na.kmeans(x, 3), na.ward(x, 3)):
        assert na.adjusted_rand(labels, truth) == pytest.approx(1.0)
        assert labels[0] == 0  # numbered in order of appearance
    assert abs(na.adjusted_rand(rng.integers(3, size=300), rng.integers(3, size=300))) < 0.05


def test_alignment_and_change_measures():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 6))
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    assert na.linear_cka(x, 3 * x @ rotation) == pytest.approx(1.0)
    assert na.linear_cka(x, rng.normal(size=(40, 6))) < 0.5
    assert na.relative_change(2 * x, x) == pytest.approx(1.0)
    edges = np.array([[0, 1], [2, 3]])
    assert na.edge_distances(x, edges).shape == (2,)


def test_four_rooms_labels():
    env = FourRooms()
    dynamics = env.tabular_dynamics()
    labels = {
        label.name: label for label in four_rooms_labels(env, dynamics, goal=(9, 9), gamma=0.9)
    }
    assert labels["room"].classes[-1] == "doorway" and labels["room"].values.shape == (68,)
    assert labels["next to a wall"].values.sum() == 44
    goal = dynamics.index_of((9, 9))
    assert labels["distance to goal"].values[goal] == 0
    assert labels["optimal action"].accepted.shape == (68, 4)
    assert labels["occupancy (random walk)"].values.sum(axis=1) == pytest.approx(np.ones(68))
    assert labels["next positions"].values.shape == (68, 8)


def test_network_features_round_trip():
    from aeris.analysis.network_features import NetworkFeatures

    features = synthetic_network_features()
    again = NetworkFeatures.from_arrays(*features.to_arrays())
    assert again.conditions == features.conditions and again.layers == features.layers
    assert again.layer("F.out", 2).shape == (68, 8) and again.layer("B.out").shape == (68, 6)
    assert again.actions(0).shape == (68,) and again.reference == 0
    with pytest.raises(ValueError, match="layer 'B.out'"):
        NetworkFeatures(
            **{**vars(features), "features": {**features.features, "B.out": np.ones((3, 6))}}
        )
