from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from predictive_representations_rl.algorithms.base import EXTRACT_DIR, AlgorithmAdapter, AlgorithmSpec
from predictive_representations_rl.core.runtime import Command, Venv

if TYPE_CHECKING:
    from predictive_representations_rl.runner import ResolvedRun

REPO = "third_party/Offline_vs_Online_in_MBRL"
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

        return Command(args=tuple(args), cwd=run.root / REPO, env={"MUJOCO_GL": "egl", **run.config.environment})

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
        return Command(args=args, cwd=run.root / REPO, env={"MUJOCO_GL": "egl", **run.config.environment})

    def find_checkpoint(self, run: ResolvedRun) -> Path | None:
        checkpoint = run.native_dir / "checkpoint.ckpt"
        return checkpoint if checkpoint.exists() else None

    def experience(self, run: ResolvedRun) -> dict[str, Any]:
        return {
            "env_steps": int(run.config.budget.get("env_steps", DEFAULT_ENV_STEPS)),
            "dataset": None,
            # Dreamer trains at a replay ratio (run.train_ratio) during collection; the count is in its metrics.
            "gradient_steps": None,
        }
