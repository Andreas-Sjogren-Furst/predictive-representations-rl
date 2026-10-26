import copy

import numpy as np
import pytest
import torch

from aeris.agents.fb import DiscreteFBAgent, DiscreteFBConfig, FBTrainingBatch, FBTransitionSampler
from aeris.contracts.agent import Agent, Checkpointable, TaskConditioner, TrainableAgent
from aeris.contracts.dataset import BatchProvider
from aeris.domain.agent import ActionRequest
from aeris.domain.experience import Transition
from aeris.domain.task import TaskSpec


def agent(*, mode="uniform_ball", seed=2):
    return DiscreteFBAgent(
        DiscreteFBConfig(
            observation_dim=4,
            num_actions=3,
            representation_dim=5,
            hidden_dim=16,
            task_sampling=mode,
            seed=seed,
            temperature=1.5,
            exploration_epsilon=0.0,
        )
    )


def batch(n=8, *, goals=False):
    gen = np.random.default_rng(7)
    a = gen.normal(size=(n, 4)).astype(np.float32)
    nxt = gen.normal(size=(n, 4)).astype(np.float32)
    oth = gen.normal(size=(n, 4)).astype(np.float32)
    actions = gen.integers(3, size=n)
    return FBTrainingBatch(a, actions, nxt, oth, a.copy() if goals else None)


def test_structural_contracts():
    a = agent()
    assert isinstance(a, Agent)
    assert isinstance(a, TrainableAgent)
    assert isinstance(a, Checkpointable)
    assert isinstance(a, TaskConditioner)


def test_shapes_and_no_implicit_learning():
    a = agent()
    before = {key: t.clone() for key, t in a.forward_network.state_dict().items()}
    z = a.goal_condition(np.ones(4, dtype=np.float32))
    out = a.act(ActionRequest(np.ones(4), condition=z))
    assert 0 <= out.action < 3
    assert a.action_values(np.ones(4), z).shape == (1, 3)
    assert a.forward_features(np.ones(4), [out.action], z).shape == (1, 5)
    assert a.backward_features(np.ones(4)).shape == (1, 5)
    assert all(torch.equal(t, before[key]) for key, t in a.forward_network.state_dict().items())
    with pytest.raises(ValueError):
        a.act(ActionRequest(np.ones(4)))


def test_update_finite_and_changes_model():
    a = agent()
    before = a.forward_network.forward_out.weight.detach().clone()
    result = a.update(batch())
    assert {x.name for x in result} == {
        "fb_loss",
        "fb_regularization",
        "total_loss",
        "target_policy_entropy",
    }
    assert all(np.isfinite(x.value) for x in result)
    assert a.updates == 1
    assert not torch.equal(a.forward_network.forward_out.weight, before)
    assert not any(p.requires_grad for p in a.forward_target_network.parameters())


def test_loss_matches_direct_reference_for_fixed_samples():
    a = agent()
    b = batch()
    obs = torch.as_tensor(b.observations)
    nxt = torch.as_tensor(b.next_observations)
    other = torch.as_tensor(b.comparison_observations)
    actions = torch.as_tensor(b.actions).long()
    # Replay RNG state with a forked copy to calculate the independent formula.
    state = a.rng.get_state().clone()
    w = a.sample_conditions(len(obs))
    a.rng.set_state(state)
    with torch.no_grad():
        ff = a.forward_target_network(nxt, w)
        q = torch.einsum("bda,bd->ba", ff, w)
        policy = torch.softmax(q / a.config.temperature, dim=-1)
        ft = torch.einsum("bda,ba->bd", ff, policy)
        target = ft @ a.backward_target_network(other).T
        f = a.forward_network(obs, w).gather(2, actions[:, None, None].expand(-1, 5, 1)).squeeze(-1)
        backwards = a.backward_network(obs)
        comparison = a.backward_network(other)
        fb = (
            0.5 * (f @ comparison.T - a.config.gamma * target).square().mean()
            - (f * backwards).sum(-1).mean()
        )
        coupling = backwards @ comparison.T
        reg = ((backwards * backwards).sum(-1)[None, :] * coupling).mean() - coupling.mean()
        expected = fb + a.config.regularization_coefficient * reg
    output = a.update(b)
    measured = {m.name: m.value for m in output}
    assert measured["fb_loss"] == pytest.approx(float(fb), abs=1e-6)
    assert measured["fb_regularization"] == pytest.approx(float(reg), abs=1e-6)
    assert measured["total_loss"] == pytest.approx(float(expected), abs=1e-6)


def test_goal_condition_requires_goals_during_update():
    a = agent(mode="goal_oriented")
    with pytest.raises(ValueError, match="goal observations"):
        a.update(batch())
    assert all(np.isfinite(m.value) for m in a.update(batch(goals=True)))


def test_training_reproducible_and_checkpoint_continuation():
    x, y = agent(seed=11), agent(seed=11)
    b = batch()
    x.update(b)
    y.update(b)
    for p, q in zip(x.forward_network.parameters(), y.forward_network.parameters(), strict=True):
        assert torch.equal(p, q)
    snap = copy.deepcopy(x.state_dict())
    z = agent(seed=11)
    z.load_state_dict(snap)
    r1, r2 = x.update(b), z.update(b)
    assert [(m.name, m.value) for m in r1] == [(m.name, m.value) for m in r2]
    for p, q in zip(x.backward_network.parameters(), z.backward_network.parameters(), strict=True):
        assert torch.equal(p, q)


def test_reward_evidence_and_batch_sampler():
    a = agent()
    z = a.infer_condition(TaskSpec(task_id="test"), [(np.ones(4), 2.0), (np.zeros(4), -1.0)])
    assert z.shape == (5,)
    assert bool(torch.isfinite(z).all())
    with pytest.raises(ValueError):
        a.infer_condition(TaskSpec(task_id="test"), [])
    transition = Transition(observation=np.ones(4), action=1, next_observation=np.zeros(4))
    sampler = FBTransitionSampler([transition], batch_size=5, seed=0)
    assert isinstance(sampler, BatchProvider)
    data = sampler.next_batch()
    assert data.observations.shape == (5, 4)
    assert data.comparison_observations.shape == (5, 4)
    assert all(data.actions == 1)


def test_invalid_action_and_shape_rejected():
    a = agent()
    b = batch()
    with pytest.raises(ValueError, match="bounds"):
        a.update(
            FBTrainingBatch(
                b.observations, np.full(8, 100), b.next_observations, b.comparison_observations
            )
        )
    with pytest.raises(ValueError, match="shape"):
        a.update(
            FBTrainingBatch(
                b.observations, b.actions, b.next_observations[:, :-1], b.comparison_observations
            )
        )


def test_single_rewarded_goal_sample_infers_exactly_the_goal_condition():
    a = agent()
    goal = np.array([0.5, -1.0, 2.0, 0.0], dtype=np.float32)
    inferred = a.infer_condition(TaskSpec(task_id="goal"), [(goal, 1.0)])
    assert torch.equal(inferred, a.goal_condition(goal))


@pytest.mark.parametrize("period", [1, 40])
def test_targets_move_only_every_target_update_period(period):
    a = DiscreteFBAgent(
        DiscreteFBConfig(
            observation_dim=4,
            num_actions=3,
            representation_dim=5,
            hidden_dim=16,
            task_sampling="uniform_ball",
            seed=2,
            target_update_period=period,
        )
    )
    b = batch()

    def target_weights():
        return a.forward_target_network.forward_out.weight.detach().clone()

    for step in range(1, period + 1):
        before = target_weights()
        a.update(b)
        moved = not torch.equal(target_weights(), before)
        assert moved == (step % period == 0), step
