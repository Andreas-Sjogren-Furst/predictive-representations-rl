"""Neural architectures matching the original discrete-action FB structure."""

import torch
from torch import nn
from torch.nn import functional as F


class ForwardMap(nn.Module):
    """(observation, task) -> [B, representation_dim, num_actions]."""

    def __init__(
        self, observation_dim: int, representation_dim: int, num_actions: int, hidden_dim: int = 256
    ):
        super().__init__()
        self.representation_dim = representation_dim
        self.num_actions = num_actions
        self.fc1 = nn.Linear(observation_dim + representation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.forward_out = nn.Linear(hidden_dim, representation_dim * num_actions)

    def forward(self, observations: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        out = self.activations(observations, conditions)["out"]
        return out.reshape(-1, self.representation_dim, self.num_actions)

    def activations(
        self, observations: torch.Tensor, conditions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Every layer's output, after its ReLU: fc1, fc2, fc3 and out ([B, d * A])."""
        d = self.representation_dim
        conditions = conditions / torch.sqrt(
            1 + torch.linalg.vector_norm(conditions, dim=-1, keepdim=True).square() / d
        )
        x = torch.cat((observations, conditions), dim=-1)
        layers = {"fc1": F.relu(self.fc1(x))}
        layers["fc2"] = F.relu(self.fc2(layers["fc1"]))
        layers["fc3"] = F.relu(self.fc3(layers["fc2"]))
        layers["out"] = self.forward_out(layers["fc3"])
        return layers


class BackwardMap(nn.Module):
    """observation -> [B, representation_dim]."""

    def __init__(self, observation_dim: int, representation_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.backward_out = nn.Linear(hidden_dim, representation_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.activations(observations)["out"]

    def activations(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every layer's output, after its ReLU: fc1, fc2, fc3 and out ([B, d])."""
        layers = {"fc1": F.relu(self.fc1(observations))}
        layers["fc2"] = F.relu(self.fc2(layers["fc1"]))
        layers["fc3"] = F.relu(self.fc3(layers["fc2"]))
        layers["out"] = self.backward_out(layers["fc3"])
        return layers
