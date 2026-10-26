"""The representation record validates shapes and derives Q and F·B consistently."""

import numpy as np
import pytest

from aeris.agents.fb import FBRepresentations
from conftest import synthetic_representations as synthetic


def test_roundtrip_through_arrays():
    reps = synthetic()
    restored = FBRepresentations.from_arrays(*reps.to_arrays())
    assert restored.states == reps.states and restored.goals == reps.goals
    assert np.array_equal(restored.forward, reps.forward)


def test_derived_quantities_match_their_definitions():
    reps = synthetic()
    f, z, b = reps.forward[0], reps.goal_conditions[0], reps.backward
    assert np.allclose(reps.q_values(0)[2, 1], f[2, :, 1] @ z, atol=1e-5)
    assert np.allclose(reps.predicted_density(0)[0, 1, 2], f[0, :, 1] @ b[2], atol=1e-5)


def test_shapes_and_goals_are_validated():
    good = synthetic()
    with pytest.raises(ValueError, match="forward has shape"):
        FBRepresentations(**{**good.__dict__, "forward": good.forward[:, :2]})
    with pytest.raises(ValueError, match="goals are not states"):
        FBRepresentations(**{**good.__dict__, "goals": ((9, 9),)})
    arrays, metadata = good.to_arrays()
    with pytest.raises(ValueError, match="schema"):
        FBRepresentations.from_arrays(arrays, {**metadata, "schema_version": 2})
