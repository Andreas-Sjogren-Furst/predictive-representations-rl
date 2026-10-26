"""Trusted local checkpoint files for the discrete FB agent.

Security: torch checkpoints are pickle-capable. Only load checkpoints that
AERIS wrote locally, never uploaded or downloaded files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .agent import DiscreteFBAgent


def save_checkpoint(
    path: str | Path, agent: DiscreteFBAgent, *, sampler: Any | None = None
) -> Path:
    """Save the agent (and optionally its batch sampler) for exact resumption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"agent": agent.state_dict()}
    if sampler is not None:
        payload["sampler"] = sampler.state_dict()
    torch.save(payload, path)
    return path


def load_agent(path: str | Path, *, device: str = "cpu") -> DiscreteFBAgent:
    """Rebuild the agent from a trusted checkpoint; weights and hyperparameters are unchanged."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "agent" not in checkpoint:
        raise ValueError(f"{path}: not an AERIS FB checkpoint")
    return DiscreteFBAgent.from_state_dict(checkpoint["agent"], device=device)
