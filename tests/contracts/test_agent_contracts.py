from collections.abc import Mapping
from typing import Any

from aeris.contracts import Agent, Checkpointable, TaskConditioner, TrainableAgent
from aeris.domain import ActionRequest, ActionResult, MetricValue, TaskSpec, Transition


class FixedPolicy:
    def act(self, request: ActionRequest) -> ActionResult:
        # A policy need not inherit from the Protocol nor know the environment.
        return ActionResult(action=0, next_policy_state=request.policy_state)


class OfflineToyLearner(FixedPolicy):
    def __init__(self) -> None:
        self.updates = 0

    def update(self, batch: tuple[Transition, Transition]) -> tuple[MetricValue, ...]:
        assert len(batch) == 2
        self.updates += 1
        return (MetricValue("loss", 1.0 / self.updates, "unitless", "instant"),)

    def state_dict(self) -> Mapping[str, Any]:
        return {"updates": self.updates}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.updates = int(state["updates"])


class NoTrainingTaskAgent(FixedPolicy):
    def infer_condition(self, task: TaskSpec, evidence: list[tuple[int, float]]) -> tuple[int, ...]:
        assert task.task_id == "goal_1"
        return tuple(i for i, reward in evidence if reward > 0)


def test_minimal_policy_is_structurally_compatible():
    agent = FixedPolicy()
    assert isinstance(agent, Agent)
    assert not isinstance(agent, TrainableAgent)
    result = agent.act(ActionRequest(observation=[1.0, 2.0], policy_state="recurrent"))
    assert result.action == 0
    assert result.next_policy_state == "recurrent"


def test_learning_is_an_optional_capability_with_algorithm_specific_batch():
    learner = OfflineToyLearner()
    assert isinstance(learner, Agent)
    assert isinstance(learner, TrainableAgent)
    batch = (Transition(0, 1, 2), Transition(2, 3, 4))
    diagnostic = learner.update(batch)
    assert diagnostic[0].name == "loss"
    assert learner.updates == 1


def test_checkpoint_state_is_optional_and_not_a_serialization_format():
    original = OfflineToyLearner()
    original.update((Transition(0, 0, 1), Transition(1, 0, 2)))
    assert isinstance(original, Checkpointable)
    restored = OfflineToyLearner()
    restored.load_state_dict(original.state_dict())
    assert restored.updates == original.updates
    assert not isinstance(FixedPolicy(), Checkpointable)


def test_task_inference_does_not_require_agent_to_be_trainable():
    policy = NoTrainingTaskAgent()
    assert isinstance(policy, Agent)
    assert isinstance(policy, TaskConditioner)
    assert not isinstance(policy, TrainableAgent)
    condition = policy.infer_condition(TaskSpec("goal_1"), [(7, 1.0), (3, 0.0)])
    assert condition == (7,)
    assert policy.act(ActionRequest(0, condition=condition)).action == 0
