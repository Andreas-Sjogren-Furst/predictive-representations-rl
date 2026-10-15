import numpy as np

from rl_lab.analysis.probes import factor_columns, probe, probe_table, skip_reason

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


def test_probe_table_rows_and_skipped_factors():
    meta = toy_metadata()
    meta["reward_far_goal"] = np.zeros(600)
    representations = {"a": np.column_stack([meta["x"], meta["y"]]), "b": RNG.normal(size=(600, 3))}

    rows, skipped = probe_table(representations, meta, meta["episode"])

    assert {(row["representation"], row["factor"]) for row in rows} == {(r, f) for r in "ab" for f in ("x", "y", "room")}
    assert skipped == {"rewarded_far_goal": "constant"}
    by_key = {(row["representation"], row["factor"]): row["score"] for row in rows}
    assert by_key[("a", "x")] > 0.99 > by_key[("b", "x")]
