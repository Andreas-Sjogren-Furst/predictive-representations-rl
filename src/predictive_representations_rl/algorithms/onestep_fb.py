from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from predictive_representations_rl.algorithms.base import AlgorithmAdapter, AlgorithmSpec
from predictive_representations_rl.core.runtime import Command, Venv

if TYPE_CHECKING:
    from predictive_representations_rl.runner import ResolvedRun

REPO = "third_party/onestep-fb"
DEFAULT_TRAIN_STEPS = 1_000_000

ONESTEP_FB = AlgorithmSpec(
    name="onestep_fb",
    description="One-step Forward-Backward representations (third_party/onestep-fb).",
    modes={"offline": "dataset"},
    action_spaces=frozenset({"continuous"}),
    observation_types=frozenset({"state", "pixels"}),
    representations=("forward", "backward", "latent"),
    stateful_representation=False,
    task_specific=False,
    runtime=Venv(f"{REPO}/.venv"),
)


class OneStepFBAdapter(AlgorithmAdapter):
    spec = ONESTEP_FB

    def train_command(self, run: ResolvedRun) -> Command:
        train_steps = int(run.config.budget.get("train_steps", DEFAULT_TRAIN_STEPS))

        flags: dict[str, Any] = {
            "env_name": run.native_env_name,
            "seed": run.seed,
            "save_dir": run.native_dir,
            "wandb_run_group": "harness",
            "train_steps": train_steps,
            # main.py only checkpoints every save_interval steps; make sure the final step is saved.
            "save_interval": train_steps,
            # wandb and videos are optional in main.py; the harness reads the CSV logs instead.
            "enable_wandb": 0,
            "video_episodes": 0,
            "agent": "agents/onestep_fb.py",
            **run.config.overrides,
        }

        return Command(
            args=("main.py", *(f"--{key}={value}" for key, value in flags.items())),
            cwd=run.root / REPO,
            env={"PYTHONPATH": str(run.root / REPO), "MUJOCO_GL": "egl", **run.config.environment},
        )

    def find_checkpoint(self, run: ResolvedRun) -> Path | None:
        checkpoints = list(run.native_dir.rglob("params_*.pkl"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda path: int(re.findall(r"\d+", path.stem)[-1]))

    def experience(self, run: ResolvedRun) -> dict[str, Any]:
        return {
            "env_steps": 0,
            "dataset": run.config.dataset,
            "dataset_files": [str(Path(f).expanduser()) for f in run.env.datasets[run.config.dataset].files],
            "gradient_steps": int(run.config.budget.get("train_steps", DEFAULT_TRAIN_STEPS)),
        }
