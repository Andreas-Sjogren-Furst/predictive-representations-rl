import numpy as np
import pytest

from predictive_representations_rl.analysis.cka import cka_matrix, linear_cka

RNG = np.random.default_rng(0)


def test_cka_is_one_for_the_same_representation_up_to_rotation_scale_and_shift():
    x = RNG.normal(size=(300, 12))
    rotation, _ = np.linalg.qr(RNG.normal(size=(12, 12)))

    assert linear_cka(x, 3.0 * x @ rotation + 5.0) == pytest.approx(1.0)


def test_cka_compares_representations_of_different_widths():
    latent = RNG.normal(size=(300, 3))
    wide = latent @ RNG.normal(size=(3, 40))
    narrow = latent @ RNG.normal(size=(3, 5))
    unrelated = RNG.normal(size=(300, 5))

    assert linear_cka(wide, narrow) > 0.6
    assert linear_cka(wide, unrelated) < 0.1


def test_cka_is_symmetric_and_bounded():
    x, y = RNG.normal(size=(2, 200, 6))

    assert linear_cka(x, y) == pytest.approx(linear_cka(y, x))
    assert 0.0 <= linear_cka(x, y) <= 1.0


def test_cka_matrix():
    x = RNG.normal(size=(100, 4))
    names, matrix = cka_matrix({"a": x, "b": 2 * x, "c": RNG.normal(size=(100, 4))})

    assert names == ["a", "b", "c"]
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    np.testing.assert_allclose(matrix, matrix.T)
    assert matrix[0, 1] == pytest.approx(1.0) and matrix[0, 2] < 0.3
