import pytest

from aeris.environments import FourRooms
from aeris.environments.four_rooms import DOORWAYS, room_of
from aeris.environments.four_rooms_datasets import PRESETS, OracleDataset, oracle_dataset

ENV = FourRooms()


def starts_and_nexts(transitions):
    starts = {ENV.position_of(t.observation) for t in transitions}
    nexts = {ENV.position_of(t.next_observation) for t in transitions}
    return starts, nexts


def test_full_is_the_original_oracle_dataset():
    full = PRESETS["full"].transitions(ENV)
    original = tuple(ENV.all_transitions())
    assert len(full) == len(original) == 272
    for a, b in zip(full, original, strict=True):
        assert a.action == b.action
        assert ENV.position_of(a.observation) == ENV.position_of(b.observation)
        assert ENV.position_of(a.next_observation) == ENV.position_of(b.next_observation)
    ENV.require_full_coverage(full)


def test_every_transition_is_the_true_one():
    for spec in PRESETS.values():
        for t in spec.transitions(ENV):
            start = ENV.position_of(t.observation)
            assert ENV.position_of(t.next_observation) == ENV.next_position(start, t.action)


def test_room_subsets_keep_the_moves_that_leave_them():
    starts, nexts = starts_and_nexts(PRESETS["room_top_left"].transitions(ENV))
    assert {room_of(s) for s in starts} == {"top-left"} and len(starts) == 16
    assert nexts - starts == {(2, 5), (5, 2)}  # its doorways: next states, never starts
    starts, _ = starts_and_nexts(PRESETS["rooms_no_bottom_right"].transitions(ENV))
    assert (9, 9) not in starts and "bottom-right" not in {room_of(s) for s in starts}


def test_missing_actions_cells_and_doorway_moves():
    assert 2 not in {t.action for t in PRESETS["no_left"].transitions(ENV)}
    starts, nexts = starts_and_nexts(PRESETS["no_doorway_moves"].transitions(ENV))
    assert not (starts | nexts) & DOORWAYS
    starts, nexts = starts_and_nexts(PRESETS["no_goal"].transitions(ENV))
    assert (9, 9) not in starts and (9, 9) in nexts
    starts, _ = starts_and_nexts(PRESETS["closed_doorway"].transitions(ENV))
    assert (8, 5) not in starts and len(starts) == 67


def test_weights_repeat_pairs_and_holes_are_seeded():
    skewed = PRESETS["skew_top_left"].transitions(ENV)
    assert len(skewed) == 16 * 4 * 10 + 52 * 4
    rare = PRESETS["rare_doorways"].pairs(ENV)
    assert {copies for state, _, copies in rare if state in DOORWAYS} == {1}
    holes = PRESETS["holes_25"].pairs(ENV)
    assert len(holes) == 272 - 68 and holes == PRESETS["holes_25"].pairs(ENV)
    other = OracleDataset("other", "", holes=0.25, hole_seed=1).pairs(ENV)
    assert other != holes


def test_invalid_specs_are_refused():
    with pytest.raises(ValueError, match="rooms must be"):
        OracleDataset("x", "", rooms=("kitchen",))
    with pytest.raises(ValueError, match="weights"):
        OracleDataset("x", "", weights={"top-left": 0})
    with pytest.raises(ValueError, match="holes"):
        OracleDataset("x", "", holes=1.0)
    with pytest.raises(ValueError, match="covers no state-action"):
        OracleDataset("x", "", rooms=("top-left",), drop_actions=(0, 1, 2, 3)).pairs(ENV)
    with pytest.raises(ValueError, match="unknown dataset scenario 'nope'"):
        oracle_dataset("nope")
    assert PRESETS["no_left"].to_dict()["drop_actions"] == [2]
