import pytest

from aeris.analysis.seeds import key_metrics, summarize, summarize_groups
from aeris.domain import EvaluationRecord, MetricValue
from conftest import synthetic_suite_report, write_four_rooms_run


def test_summarize_ignores_missing_values():
    assert summarize([1.0, None, 3.0]).mean == 2.0
    assert summarize([1.0, None, 3.0]).std == pytest.approx(2**0.5)
    assert summarize([0.5]).std == 0.0 and summarize([0.5]).n == 1
    assert summarize([None]) is None


def test_key_metrics_read_evaluations_and_reports():
    evaluation = EvaluationRecord(
        "r", "c", "t", "p", 0, 2, (MetricValue("success_rate", 0.5, "f", "m"),)
    )
    metrics = key_metrics([evaluation], [synthetic_suite_report()])
    assert metrics["final evaluation · success_rate"] == 0.5
    assert metrics["zero-shot · all · dataset_rewards"] == 0.5


def test_groups_follow_the_recorded_value(tmp_path):
    rows = []
    for index, (origin, value) in enumerate([("oracle", 0.2), ("random", 0.9), ("oracle", 0.4)]):
        run = write_four_rooms_run(tmp_path / str(index), dataset_settings={"origin": origin})
        rows.append((str(index), run.read_manifest(), {"score": value}))
    groups = summarize_groups(rows, by="dataset.origin")
    assert [g.group for g in groups] == ["oracle", "random"]
    assert groups[0].runs == ("0", "2") and groups[0].metrics["score"].mean == pytest.approx(0.3)
    assert summarize_groups(rows, by="name")[0].metrics["score"].n == 3


def test_manifest_settings_by_path(tmp_path):
    manifest = write_four_rooms_run(
        tmp_path / "r", dataset_settings={"origin": "oracle"}
    ).read_manifest()
    assert manifest.setting("dataset.origin") == "oracle"
    assert manifest.setting("evaluation.max_steps") == 10
    assert manifest.setting("seed") == 42 and manifest.setting("name") == "fb_demo"
    assert manifest.setting("agent.missing") is None and manifest.setting("nothing.here") is None
