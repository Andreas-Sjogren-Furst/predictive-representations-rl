import pytest

from aeris.domain import (
    EvaluationRecord,
    EvaluationTrajectory,
    MetricValue,
    TrainingMetricRecord,
    TrajectoryStep,
)
from aeris.domain.agent import ActionResult
from aeris.environments import FourRooms
from aeris.experiments import record_policy_snapshot


class DownRight:
    def act(self, request):
        return ActionResult(action=3)


def test_policy_snapshot_uses_public_agent_contract():
    env = FourRooms()
    snapshot = record_policy_snapshot(
        DownRight(), env.positions, env.observation_for, condition=None
    )
    assert snapshot.states == env.positions
    assert set(snapshot.actions) == {3}


def test_plot_from_typed_records(tmp_path):
    pytest.importorskip("matplotlib")
    from aeris.visualization.four_rooms import (
        render_evaluation_success,
        render_policy_map,
        render_training_metrics,
        render_trajectory,
    )

    env = FourRooms()
    snapshot = record_policy_snapshot(
        DownRight(), env.positions, env.observation_for, condition=None
    )
    assert render_policy_map(snapshot, (9, 9), tmp_path / "policy.png").stat().st_size > 1000
    trace = EvaluationTrajectory(
        run_id="r",
        task_id="t",
        checkpoint_id="initial_in_memory",
        protocol_id="p",
        seed=0,
        episode_index=0,
        initial_observation=env.observation_for((1, 1)).tolist(),
        steps=(
            TrajectoryStep(
                env.observation_for((1, 1)).tolist(),
                3,
                env.observation_for((1, 2)).tolist(),
                None,
                False,
                False,
                False,
            ),
        ),
        success=False,
    )
    assert render_trajectory(trace, (9, 9), tmp_path / "trajectory.png").stat().st_size > 1000
    loss = [
        TrainingMetricRecord(
            "r", "gradient_updates", step, (MetricValue("fb_loss", step / 100, "u", "i"),)
        )
        for step in (10, 20)
    ]
    assert render_training_metrics(loss, tmp_path / "training.png").stat().st_size > 1000
    evaluation = EvaluationRecord(
        "r",
        "checkpoint.pt",
        "t",
        "p",
        0,
        4,
        (MetricValue("success_rate", 0.25, "fraction", "mean"),),
    )
    chart = render_evaluation_success([evaluation], tmp_path / "evaluation.png")
    assert chart.stat().st_size > 1000
