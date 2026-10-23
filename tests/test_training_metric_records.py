import pytest

from aeris.domain import MetricValue, TrainingMetricRecord


def test_training_record_preserves_axis_and_serializes():
    record = TrainingMetricRecord(
        "run-1",
        "environment_steps",
        40,
        (MetricValue("collection_rate", 2.5, "steps_per_second", "instant"),),
    )
    assert record.to_dict()["step_axis"] == "environment_steps"
    assert record.to_dict()["metrics"][0]["unit"] == "steps_per_second"


@pytest.mark.parametrize("step", [-1, True, 1.0])
def test_invalid_step_rejected(step):
    with pytest.raises(ValueError):
        TrainingMetricRecord("run-1", "gradient_updates", step, ())


def test_duplicate_metrics_rejected():
    metric = MetricValue("loss", 1, "unitless", "instant")
    with pytest.raises(ValueError):
        TrainingMetricRecord("run-1", "gradient_updates", 0, (metric, metric))
