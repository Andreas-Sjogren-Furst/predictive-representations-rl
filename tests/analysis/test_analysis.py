import csv
import json

import numpy as np
import pytest

from predictive_representations_rl.analysis.base import factor_columns, load_extraction
from predictive_representations_rl.analysis.linear_probe import LinearProbeAnalyzer, probe, skip_reason
from predictive_representations_rl.analysis.pca import PCAAnalyzer, participation_ratio, pca

RNG = np.random.default_rng(0)


def toy_metadata(num=600):
    x, y = RNG.uniform(-0.3, 0.3, (2, num))
    return {
        "x": x,
        "y": y,
        "room": (x >= 0).astype(np.int64) + 2 * (y < 0).astype(np.int64),
        "episode": np.repeat(np.arange(num // 20), 20),
        "source_index": np.arange(num),
    }


def test_pca_finds_a_two_dimensional_subspace():
    latent = RNG.normal(size=(500, 2)) * [3.0, 1.0]
    values = latent @ RNG.normal(size=(2, 10))

    scores, ratio = pca(values)

    assert ratio[:2].sum() == pytest.approx(1.0)
    assert ratio[0] > ratio[1]
    assert 1.0 < participation_ratio(ratio) < 2.0
    score_variance = np.var(scores[:, :2], axis=0)
    np.testing.assert_allclose(score_variance / score_variance.sum(), ratio[:2])


def test_linear_probe_scores_signal_high_and_noise_low():
    meta = toy_metadata()
    linear = np.column_stack([meta["x"], meta["y"]]) @ RNG.normal(size=(2, 8))
    noise = RNG.normal(size=(600, 8))

    assert probe(linear, meta["x"], "continuous", meta["episode"])["score"] > 0.99
    assert probe(noise, meta["x"], "continuous", meta["episode"])["score"] < 0.05
    room = probe(linear, meta["room"], "categorical", meta["episode"])
    assert room["metric"] == "balanced_accuracy" and room["score"] > 0.95 and room["chance"] == 0.25


def test_factor_columns_skip_bookkeeping_and_detect_classes():
    factors = factor_columns(toy_metadata())

    assert set(factors) == {"x", "y", "room"}
    assert factors["room"][1] == "categorical" and factors["x"][1] == "continuous"


def test_rewards_become_binary_rewarded_labels():
    reward = np.array([0.0, 1e-300, 0.002, 0.9])

    factors = factor_columns({"reward_reach_goal": reward})

    target, kind = factors["rewarded_reach_goal"]
    assert kind == "categorical"
    np.testing.assert_array_equal(target, [0, 0, 1, 1])


def test_rare_or_constant_factors_are_skipped():
    episodes = np.repeat(np.arange(20), 10)
    rare = np.zeros(200, np.int64)
    rare[:15] = 1  # 15 positives from 2 episodes

    assert "class 1 has 15 probes from 2 episodes" == skip_reason(rare, "categorical", episodes)
    assert skip_reason(np.zeros(200), "continuous", episodes) == "constant"
    assert skip_reason(np.arange(200) % 2, "categorical", episodes) is None


def test_load_extraction_keeps_only_per_probe_arrays(tmp_path):
    meta = toy_metadata(40)
    np.savez(tmp_path / "metadata.npz", **meta)
    np.savez(tmp_path / "backward.npz", values=np.ones((40, 3)))
    np.savez(tmp_path / "latent.npz", values=np.ones((4, 3)), tasks=np.array(list("abcd")))

    representations, metadata = load_extraction(tmp_path)

    assert list(representations) == ["backward"]
    assert set(metadata) == set(meta)


def test_analyzers_write_their_outputs(tmp_path):
    meta = toy_metadata()
    representations = {"rep": np.column_stack([meta["x"], meta["y"], RNG.normal(size=600)])}

    pca_summary = PCAAnalyzer().run(representations, meta, tmp_path / "pca")
    probe_summary = LinearProbeAnalyzer().run(representations, meta, tmp_path / "linear_probe")

    assert {p.name for p in (tmp_path / "pca").iterdir()} == {"rep.npz", "map_rep.png", "explained_variance.png", "summary.json"}
    assert pca_summary["rep"]["dims"] == 3
    rows = list(csv.DictReader(open(tmp_path / "linear_probe" / "linear_probe.csv")))
    assert {row["factor"] for row in rows} == {"x", "y", "room"}
    assert probe_summary["rep"]["x"] > 0.99
    assert (tmp_path / "linear_probe" / "linear_probe.png").exists()
    assert json.loads((tmp_path / "linear_probe" / "summary.json").read_text()) == probe_summary
