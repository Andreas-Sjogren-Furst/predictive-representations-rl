"""Seed sweeps: the same Four Rooms experiment trained and analyzed for several seeds.

Each seed becomes one run directory ``<root>/<name>_seed<k>``, trained by
``aeris train four-rooms`` and then analyzed (tabular policy, zero-shot suite,
representations, and the FB occupancy diagnostic for oracle data). Seeds run
one after another locally, or in parallel as an LSF job array.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from aeris.workflows.four_rooms_training import run_name

LSF_PROFILE = "/lsf/conf/profile.lsf"


def parse_seeds(text: str) -> tuple[int, ...]:
    """``"0-4"``, ``"0,3,7"`` or a mix like ``"0-2,10"`` → distinct seeds in order."""
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            raise ValueError(f"invalid seeds {text!r}: use e.g. 0-4 or 0,3,7")
        low, high = int(match[1]), int(match[2] or match[1])
        if high < low:
            raise ValueError(f"invalid seed range {part!r}")
        seeds.extend(range(low, high + 1))
    return tuple(dict.fromkeys(seeds))


@dataclass(frozen=True)
class SeedSweep:
    name: str
    seeds: tuple[int, ...]
    train_args: tuple[str, ...] = ()  # extra options for `aeris train four-rooms`
    root: Path = Path("runs")
    python: str = sys.executable

    def run_dir(self, seed: int) -> Path:
        return self.root / run_name(self.name, seed)

    @property
    def oracle_data(self) -> bool:
        return _option_value(self.train_args, "--dataset", default="oracle") == "oracle"

    def commands(self, seed: int) -> list[list[str]]:
        """Train one seed, then run every analysis that applies to it."""
        aeris = [self.python, "-m", "aeris.cli.main"]
        run = str(self.run_dir(seed))
        steps = [
            [
                *aeris,
                "train",
                "four-rooms",
                "--name",
                self.name,
                "--seed",
                str(seed),
                "--output",
                run,
                *self.train_args,
            ],
            [*aeris, "analysis", "four-rooms", run, "--no-plots"],
            [*aeris, "analysis", "zero-shot", run],
            [*aeris, "analysis", "representations", run],
        ]
        if self.oracle_data:
            steps.append([*aeris, "analysis", "fb-checkpoint", run, "--no-plots"])
        return steps

    def run_locally(self) -> dict[int, int]:
        """Run every seed in turn; returns the exit code per seed (0 = all steps succeeded)."""
        results = {}
        for seed in self.seeds:
            code = 0
            for command in self.commands(seed):
                code = subprocess.call(command)
                if code != 0:
                    break
            results[seed] = code
        return results

    def lsf_script(
        self,
        *,
        queue: str = "hpc",
        walltime: str = "4:00",
        memory: str = "4GB",
        log_dir: Path = Path("logs/sweeps"),
        workdir: Path | None = None,
    ) -> str:
        """An LSF job-array script with one array element per seed.

        Everything is written into the script, because DTU's LSF does not pass
        the submitting shell's environment to the job.
        """
        workdir = (workdir or Path.cwd()).resolve()
        log_dir = (workdir / log_dir).resolve() if not log_dir.is_absolute() else log_dir
        cases = []
        for index, seed in enumerate(self.seeds, start=1):
            chained = " && \\\n     ".join(shlex.join(command) for command in self.commands(seed))
            cases.append(f"  {index}) # seed {seed}\n     {chained} ;;")
        return "\n".join(
            [
                "#!/bin/bash",
                f"#BSUB -J {self.name}[1-{len(self.seeds)}]",
                f"#BSUB -q {queue}",
                "#BSUB -n 1",
                '#BSUB -R "span[hosts=1]"',
                f'#BSUB -R "rusage[mem={memory}]"',
                f"#BSUB -W {walltime}",
                f"#BSUB -o {log_dir}/{self.name}_%J_%I.out",
                f"#BSUB -e {log_dir}/{self.name}_%J_%I.out",
                "set -euo pipefail",
                f"cd {shlex.quote(str(workdir))}",
                "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
                'case "$LSB_JOBINDEX" in',
                *cases,
                '  *) echo "no seed for array index $LSB_JOBINDEX" >&2; exit 1 ;;',
                "esac",
                "",
            ]
        )

    def submit_lsf(self, script: str, *, script_path: Path) -> str:
        """Save the script and submit it with bsub; returns bsub's message."""
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script, encoding="utf-8")
        log_dir = re.search(r"#BSUB -o (\S+)/", script)
        if log_dir:
            Path(log_dir[1]).mkdir(parents=True, exist_ok=True)
        # Load the LSF profile when the shell has not, as the cluster's scripts do.
        load_profile = f'[ -n "${{LSF_ENVDIR:-}}" ] || source {LSF_PROFILE}'
        shell = f"{load_profile}; bsub < {shlex.quote(str(script_path))}"
        completed = subprocess.run(
            ["bash", "-c", shell], capture_output=True, text=True, env=os.environ.copy()
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"bsub failed: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        return completed.stdout.strip()


def _option_value(args: Sequence[str], flag: str, *, default: str) -> str:
    """The value given for ``flag`` as ``--flag value`` or ``--flag=value`` (last one wins)."""
    value = default
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            value = args[index + 1]
        elif arg.startswith(f"{flag}="):
            value = arg.split("=", 1)[1]
    return value


def sweep_summary(sweep: SeedSweep) -> Sequence[str]:
    lines = [f"Sweep {sweep.name}: seeds {', '.join(map(str, sweep.seeds))}"]
    lines += [f"  seed {seed}: {sweep.run_dir(seed)}" for seed in sweep.seeds]
    return lines
