import numpy as np
import pytest

from predictive_representations_rl.analysis.spectrum import participation_ratio, pca, pca_summary

RNG = np.random.default_rng(0)


def test_pca_finds_a_two_dimensional_subspace():
    latent = RNG.normal(size=(500, 2)) * [3.0, 1.0]
    values = latent @ RNG.normal(size=(2, 10))

    scores, ratio = pca(values)

    assert ratio[:2].sum() == pytest.approx(1.0)
    assert ratio[0] > ratio[1]
    assert 1.0 < participation_ratio(ratio) < 2.0
    score_variance = np.var(scores[:, :2], axis=0)
    np.testing.assert_allclose(score_variance / score_variance.sum(), ratio[:2])


def test_participation_ratio_bounds():
    assert participation_ratio(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert participation_ratio(np.full(8, 1 / 8)) == pytest.approx(8.0)
    assert participation_ratio(np.zeros(3)) == 0.0


def test_pca_summary():
    values = np.column_stack([RNG.normal(size=400) * 10, RNG.normal(size=400), np.zeros(400)])

    summary, scores, ratio = pca_summary(values)

    assert summary["dims"] == 3
    assert summary["components_for_90pct"] == 1 and summary["components_for_95pct"] == 1
    assert 1.0 < summary["participation_ratio"] < 1.1
    assert len(summary["explained_variance_ratio_top10"]) == 3
    assert scores.shape == (400, 3) and ratio.shape == (3,)
