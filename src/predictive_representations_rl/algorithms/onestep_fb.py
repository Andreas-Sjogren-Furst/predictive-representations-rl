from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from predictive_representations_rl.algorithms.base import EXTRACT_DIR, AlgorithmAdapter, AlgorithmSpec
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
    representations=("forward", "backward", "latent", "q_values"),
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
            env=self._env(run),
        )

    def extract_command(self, run: ResolvedRun, checkpoint: Path, probes: Path, out_dir: Path) -> Command:
        if run.config.extract:
            raise ValueError(f"onestep_fb extraction takes no options, got {sorted(run.config.extract)}")
        script = EXTRACT_DIR / "onestep_fb_extract.py"
        args = (str(script), "--checkpoint", str(checkpoint), "--probes", str(probes), "--out", str(out_dir))
        return Command(args=args, cwd=run.root / REPO, env=self._env(run))

    def _env(self, run: ResolvedRun) -> dict[str, str]:
        return {"PYTHONPATH": str(run.root / REPO), "MUJOCO_GL": "egl", **run.config.environment}

    def find_checkpoint(self, run: ResolvedRun) -> Path | None:
        checkpoints = list(run.native_dir.rglob("params_*.pkl"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda path: int(re.findall(r"\d+", path.stem)[-1]))

    def checkpoint_step(self, run: ResolvedRun, checkpoint: Path) -> tuple[int, str]:
        return int(re.findall(r"\d+", checkpoint.stem)[-1]), "gradient_steps"

    def view_protocol(self, run: ResolvedRun, view: str, info: dict[str, Any]) -> dict[str, Any]:
        base = view.removesuffix("_untrained")
        protocol: dict[str, Any] = {
            "input": "final_state_and_action",
            "conditioning": {"type": "none"},
            "stochastic": False,
            "note": "final state of each probe window with the dataset action taken there, clipped as in training",
        }
        if base == "forward":
            protocol["note"] += "; mean over the forward ensemble"
        if base == "q_values":
            protocol["conditioning"] = {
                "type": "inferred_z_per_task",
                "tasks": info.get("tasks", []),
                "source": "z inferred per task from reward-labelled validation data, as main.py does; Q = F(s,a)·z",
            }
        return protocol

    def budget(self, run: ResolvedRun) -> dict[str, int | None]:
        return {"gradient_steps": int(run.config.budget.get("train_steps", DEFAULT_TRAIN_STEPS)), "env_steps": 0}

    def native_metrics(self, run: ResolvedRun) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        import csv

        training, evaluation = [], []
        for path in run.native_dir.rglob("train.csv"):
            for row in csv.DictReader(open(path)):
                step = int(float(row.pop("step")))
                for metric, value in row.items():
                    if value not in ("", None):
                        training.append({"training_step": step, "step_unit": "gradient_steps", "metric": metric, "value": float(value)})
        for path in run.native_dir.rglob("eval.csv"):
            for row in csv.DictReader(open(path)):
                step = int(float(row.pop("step")))
                for column, value in row.items():
                    match = re.fullmatch(r"evaluation/(.+)_episode\.return", column)
                    if match and match.group(1) != "overall" and value not in ("", None):
                        evaluation.append({"training_step": step, "step_unit": "gradient_steps", "task": match.group(1),
                                           "metric": "episode_return", "value": float(value)})
        return training, evaluation

    def experience(self, run: ResolvedRun) -> dict[str, Any]:
        return {
            "env_steps": 0,
            "dataset": run.config.dataset,
            "dataset_files": [str(Path(f).expanduser()) for f in run.env.datasets[run.config.dataset].files],
            "gradient_steps": int(run.config.budget.get("train_steps", DEFAULT_TRAIN_STEPS)),
        }
