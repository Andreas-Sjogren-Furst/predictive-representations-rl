"""AERIS adaptation of Touati & Ollivier (2021) discrete-action FB.

Reference: github.com/ahmed-touati/controllable_agent/grid_modules/fb_agent.py
Preserves the all-action forward map, Boltzmann target expectation, FB loss,
partially detached orthonormality regularizer, and Polyak target updates.
Unlike the reference, this object does not own an environment or replay buffer.
"""
from __future__ import annotations

from dataclasses import asdict
from collections.abc import Mapping, Sequence
from typing import Any
import math

import numpy as np
import torch
from torch.nn import functional as F

from aeris.domain.agent import ActionRequest, ActionResult
from aeris.domain.results import MetricValue
from aeris.domain.task import TaskSpec
from .config import DiscreteFBConfig
from .batch import FBTrainingBatch
from .networks import ForwardMap, BackwardMap


class DiscreteFBAgent:
    """Discrete-action Forward-Backward agent implementing AERIS protocols."""

    def __init__(self, config: DiscreteFBConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Isolate initialization from the global torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.forward_network = ForwardMap(config.observation_dim, config.representation_dim, config.num_actions, config.hidden_dim).to(self.device)
            self.backward_network = BackwardMap(config.observation_dim, config.representation_dim, config.hidden_dim).to(self.device)
            self.forward_target_network = ForwardMap(config.observation_dim, config.representation_dim, config.num_actions, config.hidden_dim).to(self.device)
            self.backward_target_network = BackwardMap(config.observation_dim, config.representation_dim, config.hidden_dim).to(self.device)

        self.forward_target_network.load_state_dict(self.forward_network.state_dict())
        self.backward_target_network.load_state_dict(self.backward_network.state_dict())
        self.forward_target_network.requires_grad_(False)
        self.backward_target_network.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            list(self.forward_network.parameters()) + list(self.backward_network.parameters()),
            lr=config.learning_rate,
        )

        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(config.seed + 1)

        self.updates = 0

    def _observations(self, value: Any, *, label: str = "observations") -> torch.Tensor:
        data = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if data.ndim == 1:
            data = data.unsqueeze(0)

        if data.ndim != 2 or data.shape[1] != self.config.observation_dim:
            raise ValueError(f"{label} must have shape [B, {self.config.observation_dim}]")
        
        if not bool(torch.isfinite(data).all()):
            raise ValueError(f"{label} must be finite")
        
        return data

    def _conditions(self, value: Any, *, batch_size: int) -> torch.Tensor:
        z = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if z.ndim == 1:
            z = z.unsqueeze(0)

        if z.shape != (batch_size, self.config.representation_dim):
            # Single condition broadcasts across an observation batch.
            if z.shape == (1, self.config.representation_dim):
                z = z.expand(batch_size, -1)
            else:
                raise ValueError(f"condition must have shape [B, {self.config.representation_dim}] or [{self.config.representation_dim}]")
            
        if not bool(torch.isfinite(z).all()):
            raise ValueError("condition must be finite")
        
        return z

    @torch.no_grad()
    def sample_conditions(self, count: int, *, goals: Any | None = None) -> torch.Tensor:
        if count < 1:
            raise ValueError("count must be positive")
        mode, d = self.config.task_sampling, self.config.representation_dim
        if mode == "goal_oriented":
            if goals is None:
                raise ValueError("goal_oriented sampling requires goal observations in the training batch")
            obs = self._observations(goals, label="goal observations")
            if len(obs) != count:
                raise ValueError("goal count must match batch size")
            return self.backward_network(obs).detach()
        directions = torch.randn((count, d), generator=self.rng)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        u = torch.rand((count, 1), generator=self.rng)
        if mode == "uniform_ball":
            radii = u
        elif mode == "cauchy_ball":
            # Inverse CDF of Cauchy(0, 0.5), as in upstream.
            radii = 0.5 * torch.tan(math.pi * (u.clamp(1e-7, 1 - 1e-7) - 0.5))
        else:
            raise ValueError(f"unknown task sampling: {mode}")
        return (math.sqrt(d) * directions * radii).to(self.device)

    @torch.no_grad()
    def action_values(self, observations: Any, condition: Any) -> torch.Tensor:
        obs = self._observations(observations)
        z = self._conditions(condition, batch_size=len(obs))
        f = self.forward_network(obs, z)
        return torch.einsum("bda,bd->ba", f, z)

    @torch.no_grad()
    def act(self, request: ActionRequest) -> ActionResult:
        if request.policy_state is not None:
            raise ValueError("discrete FB has no recurrent policy state")
        if request.condition is None:
            raise ValueError("FB action selection requires a task condition")
        obs = self._observations(request.observation)
        if len(obs) != 1:
            raise ValueError("act() expects one observation, not a batch")
        values = self.action_values(obs, request.condition)
        greedy = int(values.argmax(dim=-1).item())
        if not request.deterministic and torch.rand((), generator=self.rng).item() < self.config.exploration_epsilon:
            greedy = int(torch.randint(self.config.num_actions, (), generator=self.rng).item())
        return ActionResult(action=greedy)

    @torch.no_grad()
    def goal_condition(self, goal_observation: Any) -> torch.Tensor:
        """Goal-style task condition B(goal), from the original grid-world code."""
        obs = self._observations(goal_observation, label="goal observation")
        if len(obs) != 1:
            raise ValueError("expected one goal observation")
        return self.backward_network(obs).squeeze(0).clone()

    @torch.no_grad()
    def infer_condition(self, task: TaskSpec, evidence: Sequence[tuple[Any, float]]) -> torch.Tensor:
        """Reward-weighted B features from pre-authorized state/reward pairs.

        The task supplies identity only; the caller must ensure evidence comes
        from an allowed task-inference split, not evaluation rollouts.
        """
        del task
        if not evidence:
            raise ValueError("nonempty reward evidence is required")
        obs = self._observations(np.stack([np.asarray(pair[0], dtype=np.float32) for pair in evidence]))
        rewards = torch.as_tensor([pair[1] for pair in evidence], dtype=torch.float32, device=self.device)
        if not bool(torch.isfinite(rewards).all()):
            raise ValueError("reward evidence must be finite")
        return (self.backward_network(obs) * rewards.unsqueeze(-1)).mean(dim=0).detach()

    @torch.no_grad()
    def forward_features(self, observations: Any, actions: Any, condition: Any) -> torch.Tensor:
        """Inspect selected forward vectors, shape [B, d]."""
        obs = self._observations(observations)
        z = self._conditions(condition, batch_size=len(obs))
        acts = self._actions(actions, len(obs))
        return self.forward_network(obs, z).gather(
            dim=2, index=acts[:, None, None].expand(-1, self.config.representation_dim, 1)
        ).squeeze(-1).clone()

    @torch.no_grad()
    def backward_features(self, observations: Any) -> torch.Tensor:
        return self.backward_network(self._observations(observations)).clone()

    def _actions(self, actions: Any, count: int) -> torch.Tensor:
        a = torch.as_tensor(actions, device=self.device)
        if a.ndim != 1 or len(a) != count:
            raise ValueError("actions must be [B]")
        if a.dtype == torch.bool or not (torch.is_floating_point(a) or a.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError("actions must be integer indices")
        if torch.is_floating_point(a) and not bool(torch.all(a == torch.floor(a))):
            raise ValueError("actions must be integer indices")
        if not bool(torch.isfinite(a.float()).all()) or not bool(((a >= 0) & (a < self.config.num_actions)).all()):
            raise ValueError("action index out of bounds")
        return a.long()

    def update(self, batch: FBTrainingBatch) -> tuple[MetricValue, ...]:
        obs = self._observations(batch.observations)
        nxt = self._observations(batch.next_observations, label="next_observations")
        other = self._observations(batch.comparison_observations, label="comparison_observations")
        n = len(obs)
        if n != len(nxt) or n != len(other):
            raise ValueError("all three observation batches must have the same size")
        actions = self._actions(batch.actions, n)
        w = self.sample_conditions(n, goals=batch.goal_observations)
        d = self.config.representation_dim

        # Reference: policy induced by the *target* forward network.
        with torch.no_grad():
            target_next = self.forward_target_network(nxt, w)
            q_next = torch.einsum("bda,bd->ba", target_next, w)
            pi = F.softmax(q_next / self.config.temperature, dim=-1)
            entropy = -(pi * pi.clamp_min(1e-30).log()).sum(dim=-1).mean()
            f_next = torch.einsum("bda,ba->bd", target_next, pi)
            b_other_target = self.backward_target_network(other)
            target_successor = f_next @ b_other_target.T

        f = self.forward_network(obs, w).gather(2, actions[:, None, None].expand(-1, d, 1)).squeeze(-1)
        b = self.backward_network(obs)
        b_other = self.backward_network(other)
        successor = f @ b_other.T
        diagonal = (f * b).sum(dim=-1)
        fb_loss = 0.5 * (successor - self.config.gamma * target_successor).square().mean() - diagonal.mean()

        # Intentionally asymmetric detach placement from original FB grid loss.
        b_b_other = b @ b_other.T
        b_b_other_detach = b @ b_other.detach().T
        b_b_detach = (b * b.detach()).sum(dim=-1)
        # Upstream multiplies a [B] vector by a [B,B] matrix: broadcasting
        # aligns the vector to the *columns* (not the rows).
        reg_loss = (b_b_detach[None, :] * b_b_other.detach()).mean() - b_b_other_detach.mean()
        total_loss = fb_loss + self.config.regularization_coefficient * reg_loss
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("FB loss became nonfinite")
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self._polyak_update(self.forward_target_network, self.forward_network)
            self._polyak_update(self.backward_target_network, self.backward_network)
        self.updates += 1
        return (
            MetricValue(name="fb_loss", value=float(fb_loss.detach()), unit="unitless", aggregation="instant"),
            MetricValue(name="fb_regularization", value=float(reg_loss.detach()), unit="unitless", aggregation="instant"),
            MetricValue(name="total_loss", value=float(total_loss.detach()), unit="unitless", aggregation="instant"),
            MetricValue(name="target_policy_entropy", value=float(entropy.detach()), unit="nats", aggregation="mean"),
        )

    def _polyak_update(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        for t, s in zip(target.parameters(), source.parameters()):
            t.mul_(self.config.polyak).add_(s, alpha=1.0 - self.config.polyak)

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "forward": self.forward_network.state_dict(),
            "backward": self.backward_network.state_dict(),
            "forward_target": self.forward_target_network.state_dict(),
            "backward_target": self.backward_target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "torch_generator": self.rng.get_state(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["config"] != asdict(self.config):
            raise ValueError("FB checkpoint configuration differs from agent configuration")
        self.forward_network.load_state_dict(state["forward"])
        self.backward_network.load_state_dict(state["backward"])
        self.forward_target_network.load_state_dict(state["forward_target"])
        self.backward_target_network.load_state_dict(state["backward_target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.set_state(state["torch_generator"])
        self.updates = int(state["updates"])
