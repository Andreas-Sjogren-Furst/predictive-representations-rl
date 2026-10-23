import numpy as np
import pytest

from aeris.analysis.tabular_occupancy import exact_discounted_occupancy, centered_pearson


def test_one_state_self_loop_has_geometric_visitation():
    t = np.array([[0, 0]], dtype=np.int64)
    pi = np.array([[0.25, 0.75]], dtype=np.float64)
    m = exact_discounted_occupancy(t, pi, gamma=0.8)
    assert m.shape == (1, 2, 1)
    assert np.allclose(m, 5.0)


def test_two_state_chain_has_correct_analytic_first_action():
    # From state 0, action 0 goes to absorbing state 1. Action 1 stays at 0.
    # Regardless of pi at state 0, current-state occupancy contains an
    # initial hit at 0, while next-step occupancy is policy-dependent.
    t = np.array([[1, 0], [1, 1]])
    pi = np.array([[1.0, 0.0], [1.0, 0.0]])
    m = exact_discounted_occupancy(t, pi, gamma=0.5)
    assert m[0, 0, 0] == pytest.approx(1.0)
    assert m[0, 0, 1] == pytest.approx(1.0)
    assert m[0, 1, 0] == pytest.approx(1.5)
    assert m[0, 1, 1] == pytest.approx(0.5)
    assert np.allclose(m.sum(axis=2), 2.0)


def test_stochastic_policy_and_ill_formed_inputs():
    t = np.array([[0, 1], [0, 1]])
    pi = np.ones((2, 2)) / 2
    m = exact_discounted_occupancy(t, pi, gamma=.9)
    assert np.allclose(m.sum(axis=-1), 10)
    with pytest.raises(ValueError, match='sum to one'):
        exact_discounted_occupancy(t, pi * 2, gamma=.9)
    with pytest.raises(ValueError, match='next-state'):
        exact_discounted_occupancy(np.array([[3, 1], [0, 1]]), pi, gamma=.9)
    with pytest.raises(ValueError, match='gamma'):
        exact_discounted_occupancy(t, pi, gamma=1)


def test_pearson_scale_invariance_and_degeneracy():
    x = np.array([-1, 0, 1])
    assert centered_pearson(x, 8 * x + 10) == pytest.approx(1)
    assert centered_pearson(x, -x) == pytest.approx(-1)
    assert centered_pearson(x, np.ones(3)) is None
