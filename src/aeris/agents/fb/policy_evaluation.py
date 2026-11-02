"""Learn the successor measure of one fixed policy on a frozen B (needs torch).

``FBPolicyEvaluator`` trains F^pi so that F^pi(s, a) · B(s') rho(s') estimates
the policy's discounted occupancy M^pi(s, a, s'), with FB's own loss and target
networks. Two things differ from FB training: B is the trained agent's, frozen,
and the TD target takes the fixed policy's action distribution at the next
state instead of FB's own policy. Q^pi of any reward r then follows as
F^pi(s, a) · z_r with z_r = E_rho[r B], as in FB's task inference.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from .config import DiscreteFBConfig
from .networks import BackwardMap, PolicyForwardMap


class FBPolicyEvaluator:
    def __init__(self, config: DiscreteFBConfig, backward: BackwardMap, *, seed: int = 0):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(seed)
        self.backward = copy.deepcopy(backward).to(self.device).eval()
        for parameter in self.backward.parameters():
            parameter.requires_grad_(False)
        self.forward_network = PolicyForwardMap(
            config.observation_dim, config.representation_dim, config.num_actions, config.hidden_dim
        ).to(self.device)
        self.forward_target = copy.deepcopy(self.forward_network)
        self.optimizer = torch.optim.Adam(
            self.forward_network.parameters(), lr=config.learning_rate
        )
        self.updates = 0

    def _tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def update(
        self,
        observations: Any,
        actions: Any,
        next_observations: Any,
        comparison_observations: Any,
        next_policy: Any,
    ) -> float:
        """One step on a batch; ``next_policy`` [B, A] is pi's action distribution at each
        next observation. Returns the FB loss (Bellman residual minus the diagonal term)."""
        obs, nxt = self._tensor(observations), self._tensor(next_observations)
        other = self._tensor(comparison_observations)
        pi = self._tensor(next_policy)
        index = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        with torch.no_grad():
            b_other = self.backward(other)
            f_next = torch.einsum("bda,ba->bd", self.forward_target(nxt), pi)
            target = f_next @ b_other.T
            b = self.backward(obs)
        forward = self.forward_network(obs)
        f = forward[torch.arange(len(index), device=self.device), :, index]
        successor = f @ b_other.T
        diagonal = (f * b).sum(dim=-1)
        loss = 0.5 * (successor - self.config.gamma * target).square().mean() - diagonal.mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("policy-evaluation loss became nonfinite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_update_period == 0:
            with torch.no_grad():
                for t, s in zip(
                    self.forward_target.parameters(), self.forward_network.parameters(), strict=True
                ):
                    t.mul_(self.config.polyak).add_(s, alpha=1.0 - self.config.polyak)
        return float(loss.detach())

    @torch.no_grad()
    def forward_vectors(self, observations: Any) -> torch.Tensor:
        """Learned F^pi(s, a) for every action, [B, d, A]."""
        return self.forward_network(self._tensor(observations))
