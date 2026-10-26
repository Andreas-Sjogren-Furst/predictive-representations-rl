"""evaluate_zero_shot works with any ZeroShotAgent, not just FB."""

from aeris.contracts import ZeroShotAgent
from aeris.domain import TaskEvidenceRef, TaskSpec
from aeris.domain.agent import ActionResult
from aeris.environments import FourRooms
from aeris.experiments import evaluate_zero_shot


class RecordingAgent:
    """Infers 'go right' from evidence; records what inference was shown."""

    def __init__(self):
        self.inferred_from = []
        self.conditions_seen = set()

    def infer_condition(self, task, evidence):
        self.inferred_from.append((task.task_id, list(evidence)))
        return "right"

    def act(self, request):
        self.conditions_seen.add(request.condition)
        return ActionResult(action=3)


def test_inference_happens_once_from_the_evidence_only():
    env = FourRooms(max_episode_steps=5)
    agent = RecordingAgent()
    assert isinstance(agent, ZeroShotAgent)
    goal = env.observation_for((1, 3))
    ref = TaskEvidenceRef("toy", "inference")
    record = evaluate_zero_shot(
        agent,
        env,
        task=TaskSpec("reach_1_3"),
        evidence=[("s", 1.0)],
        evidence_ref=ref,
        is_goal=lambda obs: (obs == goal).all(),
        episodes=3,
        max_steps=5,
        run_id="r",
        checkpoint_id="c",
        seed=0,
    )
    assert agent.inferred_from == [("reach_1_3", [("s", 1.0)])]
    assert agent.conditions_seen == {"right"}
    assert record.inference_evidence == ref
    assert record.protocol_id == "zero_shot_goal_reaching_v1"
    assert record.episode_count == 3
