"""Discrete-action Forward-Backward (Touati & Ollivier, 2021).

``DiscreteFBAgent`` needs the ``fb-torch`` extra and is imported on first use,
so configs, batches and reports stay usable without torch.
"""

from typing import TYPE_CHECKING, Any

from .batch import FBTrainingBatch, FBTransitionSampler
from .config import DiscreteFBConfig
from .reports import FBCheckpointReport

if TYPE_CHECKING:
    from .agent import DiscreteFBAgent

__all__ = [
    "DiscreteFBAgent",
    "DiscreteFBConfig",
    "FBCheckpointReport",
    "FBTrainingBatch",
    "FBTransitionSampler",
]


def __getattr__(name: str) -> Any:
    if name == "DiscreteFBAgent":
        from .agent import DiscreteFBAgent

        return DiscreteFBAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
