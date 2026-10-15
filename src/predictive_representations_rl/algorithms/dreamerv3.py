from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from predictive_representations_rl.algorithms.base import EXTRACT_DIR, AlgorithmAdapter, AlgorithmSpec
from predictive_representations_rl.core.runtime import Command, Venv

if TYPE_CHECKING:
    from predictive_representations_rl.runner import ResolvedRun

REPO = "third_party/Offline_vs_Online_in_MBRL"
# Shadows the repo's custom_dmc_tasks with onestep-fb's ExORL task definitions (see that package's docstring).
TASK_OVERRIDES = Path(__file__).resolve().parents[1] / "overrides" / "dreamerv3"
DEFAULT_ENV_STEPS = 250_000  # run.steps in dreamerv3/configs.yaml
# The repo's config block per suite and observation type.
CONFIGS = {("dmc", "state"): "dmc_proprio", ("dmc", "pixels"): "dmc_vision"}

DREAMERV3 = AlgorithmSpec(
    name="dreamerv3",
    description="DreamerV3 world model, online/offline fork (third_party/Offline_vs_Online_in_MBRL).",
    # online = active training; offline = passive training on another run's replay buffer.
    modes={"online": "self_collected", "offline": "replay"},
    action_spaces=frozenset({"continuous", "discrete"}),
    observation_types=frozenset({"state", "pixels"}),
    representations=("deter", "stoch", "model_state"),
    stateful_representation=True,
    task_specific=True,
    # Built by scripts/setup_dreamerv3_venv.sh (the repo itself recommends its Singularity image).
    runtime=Venv(".venvs/dreamerv3"),
)


class DreamerV3Adapter(AlgorithmAdapter):
    spec = DREAMERV3

    def train_command(self, run: ResolvedRun) -> Command:
        if run.config.mode != "online":
            raise NotImplementedError("dreamerv3 offline (passive) runs are not wired into the harness yet")

        suite_config = CONFIGS.get((run.env.suite, run.config.obs_type))
        if suite_config is None:
            raise ValueError(f"no DreamerV3 config for suite {run.env.suite!r} with {run.config.obs_type!r} observations")

        flags: dict[str, Any] = {
            "task": run.native_env_name,
            "logdir": run.native_dir,
            "seed": run.seed,
            "run.steps": int(run.config.budget.get("env_steps", DEFAULT_ENV_STEPS)),
            # Plain Dreamer, as loops_auto.sh runs it without Plan2Explore.
            "method": "pure_dreamer",
            "expl_behavior": "None",
            **run.config.overrides,
        }

        args = ["dreamerv3/train.py", "--configs", suite_config]
        for key, value in flags.items():
            args += [f"--{key}", str(value)]

        return Command(args=tuple(args), cwd=run.root / REPO, env=self._env(run))

    def extract_command(self, run: ResolvedRun, checkpoint: Path, probes: Path, out_dir: Path) -> Command:
        options = dict(run.config.extract)
        posterior = options.pop("posterior", "sample")
        if options:
            raise ValueError(f"unknown dreamerv3 extract options {sorted(options)}; supported: posterior")
        if posterior not in ("sample", "mode"):
            raise ValueError(f"extract.posterior must be 'sample' or 'mode', got {posterior!r}")

        script = EXTRACT_DIR / "dreamerv3_extract.py"
        args = (str(script), "--checkpoint", str(checkpoint), "--probes", str(probes), "--out", str(out_dir),
                "--posterior", posterior)
        return Command(args=args, cwd=run.root / REPO, env=self._env(run))

    def _env(self, run: ResolvedRun) -> dict[str, str]:
        return {"PYTHONPATH": str(TASK_OVERRIDES), "MUJOCO_GL": "egl", **run.config.environment}

    def find_checkpoint(self, run: ResolvedRun) -> Path | None:
        checkpoint = run.native_dir / "checkpoint.ckpt"
        return checkpoint if checkpoint.exists() else None

    def checkpoint_step(self, run: ResolvedRun, checkpoint: Path) -> tuple[int, str]:
        # Dreamer's logged step counts environment steps (agent decisions × action repeat).
        steps = [record["step"] for record in self._metrics(run)]
        return (int(max(steps)) if steps else 0), "env_steps"

    def view_protocol(self, run: ResolvedRun, view: str, info: dict[str, Any]) -> dict[str, Any]:
        return {
            "input": "window_filter",
            "window_length": info.get("window_length"),
            "conditioning": {"type": "none"},
            "stochastic": True,
            "posterior": info.get("posterior"),
            "sampling_noise": info.get("sampling_noise"),
            "note": "RSSM posterior filtered over the probe window from the learned initial state; a_{t-1} feeds step t; "
                    "value at the final step",
        }

    def budget(self, run: ResolvedRun) -> dict[str, int | None]:
        agent_steps = int(run.config.budget.get("env_steps", DEFAULT_ENV_STEPS))  # run.steps counts agent decisions
        repeat = self._action_repeat(run)
        return {"agent_steps": agent_steps, "env_steps": agent_steps * repeat if repeat else None, "gradient_steps": None}

    def native_metrics(self, run: ResolvedRun) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        training, evaluation = [], []
        for record in self._metrics(run):
            step = int(record["step"])
            for key, value in record.items():
                if key == "step" or not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if key == "eval_episode/score":
                    evaluation.append({"training_step": step, "step_unit": "env_steps", "task": run.config.task,
                                       "metric": "episode_return", "value": float(value)})
                else:
                    training.append({"training_step": step, "step_unit": "env_steps", "metric": key, "value": float(value)})
        return training, evaluation

    def _metrics(self, run: ResolvedRun) -> list[dict[str, Any]]:
        import json

        path = run.native_dir / "metrics.jsonl"
        return [json.loads(line) for line in open(path)] if path.exists() else []

    def _action_repeat(self, run: ResolvedRun) -> int | None:
        import yaml

        path = run.native_dir / "config.yaml"
        if not path.exists():
            return None
        config = yaml.safe_load(path.read_text())
        return config.get("env", {}).get(run.env.suite, {}).get("repeat")

    def experience(self, run: ResolvedRun) -> dict[str, Any]:
        return {
            "env_steps": int(run.config.budget.get("env_steps", DEFAULT_ENV_STEPS)),
            "dataset": None,
            # Dreamer trains at a replay ratio (run.train_ratio) during collection; the count is in its metrics.
            "gradient_steps": None,
        }
