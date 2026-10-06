"""Launch an algorithm's own training code for one (config, seed) and record the run in a standard directory.

runs/<env>/<algo>/<mode>[/<task>]/seed_<N>/
    run.json      resolved config, data source, experience budget, command, git commits, status, extractions
    train.log     stdout + stderr of the algorithm process
    native/       everything the algorithm wrote itself (logs, checkpoints, replay), untouched
    probes/<probe set>/<representation>.npz, metadata.npz, extract_info.json, extract.log
    analysis/<probe set>/<analyser>/      per-run analysis outputs

runs/<env>/probes/<probe set>.npz    shared probe inputs, built once per environment
runs/<env>/analysis/<probe set>/     cross-run comparisons
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from predictive_representations_rl.algorithms.base import AlgorithmAdapter
from predictive_representations_rl.core import registry
from predictive_representations_rl.core.compat import check
from predictive_representations_rl.core.config import ExperimentConfig
from predictive_representations_rl.envs.base import EnvSpec
from predictive_representations_rl.probes.probe_set import ProbeSet, build_from_exorl

# Where runs go unless PRL_RUNS_DIR points elsewhere (e.g. scratch space).
RUNS_DIR_VARIABLE = "PRL_RUNS_DIR"

DEFAULT_RESOURCES = {"queue": "gpua100", "cores": 4, "mem_gb": 8, "gpus": 1, "walltime": "24:00"}

DEFAULT_PROBES = {"num": 4096, "length": 16, "seed": 0}

# A run directory in this state may be started without --force: it was created by `--submit lsf` for the job.
STARTABLE = {"submitted"}


class RunError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedRun:
    """One config + one seed, with its specs looked up and its directory decided."""

    config: ExperimentConfig
    config_path: Path | None
    seed: int
    env: EnvSpec
    adapter: AlgorithmAdapter
    root: Path
    run_dir: Path

    @property
    def native_dir(self) -> Path:
        return self.run_dir / "native"

    @property
    def native_env_name(self) -> str:
        return self.env.native_names[self.adapter.spec.name](self.config.task, self.config.obs_type)


def runs_dir(root: Path) -> Path:
    return Path(os.environ.get(RUNS_DIR_VARIABLE, root / "runs")).expanduser()


def resolve(
    config: ExperimentConfig, seed: int, root: Path, config_path: Path | None = None, allow_missing: bool = False
) -> ResolvedRun:
    """Look up specs and the run directory. `allow_missing` tolerates absent datasets/runtimes (for dry runs)."""
    env = registry.get_env(config.env)
    adapter = registry.get_adapter(config.algo)

    issues = check(config, env, adapter.spec, root)
    if allow_missing:
        issues = [issue for issue in issues if issue.kind != "missing"]
    if issues:
        details = "\n".join(f"  [{issue.kind}] {issue.message}" for issue in issues)
        raise RunError(f"{config.algo} on {config.env} cannot run:\n{details}")

    run_dir = runs_dir(root) / config.env / config.algo / config.mode
    if config.task is not None:
        run_dir /= config.task
    run_dir /= f"seed_{seed}"

    return ResolvedRun(config, config_path, seed, env, adapter, root, run_dir)


def command_line(run: ResolvedRun) -> tuple[list[str], dict[str, str], Path]:
    command = run.adapter.train_command(run)
    return run.adapter.spec.runtime.argv(command, run.root), dict(command.env), command.cwd


def run_locally(run: ResolvedRun, force: bool = False) -> dict[str, Any]:
    """Run training in this process's machine and block until it finishes. Returns the final run.json."""
    _claim_run_dir(run, force)
    argv, env, cwd = command_line(run)
    record = _initial_record(run, argv, env, cwd, status="running")
    _write_record(run, record)

    started = time.monotonic()
    with open(run.run_dir / "train.log", "ab") as log:
        process = subprocess.run(argv, cwd=cwd, env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT)

    checkpoint = run.adapter.find_checkpoint(run)
    record.update(
        status="completed" if process.returncode == 0 else "failed",
        exit_code=process.returncode,
        finished_at=_now(),
        duration_s=round(time.monotonic() - started, 1),
        checkpoint=str(checkpoint) if checkpoint else None,
    )
    _write_record(run, record)
    return record


def submit_lsf(run: ResolvedRun, force: bool = False) -> str:
    """Submit an LSF job that runs `prl run` for this config and seed on a compute node. Returns the job id."""
    if run.config_path is None:
        raise RunError("LSF submission needs the run to come from a config file")

    _claim_run_dir(run, force)
    argv, env, cwd = command_line(run)
    record = _initial_record(run, argv, env, cwd, status="submitted")

    script = _lsf_script(run)
    (run.run_dir / "job.lsf").write_text(script)

    output = subprocess.run(["bsub"], input=script, text=True, capture_output=True, env=_lsf_environment())
    if output.returncode != 0:
        raise RunError(f"bsub failed: {output.stderr.strip() or output.stdout.strip()}")

    job_id = output.stdout.split("<", 1)[1].split(">", 1)[0] if "<" in output.stdout else output.stdout.strip()
    record["lsf_job_id"] = job_id
    _write_record(run, record)
    return job_id


def probe_set_path(root: Path, env: EnvSpec, name: str = "default") -> Path:
    return runs_dir(root) / env.name / "probes" / f"{name}.npz"


def build_probe_set(root: Path, env: EnvSpec, name: str = "default", force: bool = False, **params: int) -> Path:
    """Cut a probe set from the environment's held-out probe file, once; every algorithm then reuses it."""
    path = probe_set_path(root, env, name)
    if path.exists() and not force:
        return path

    if env.probe_dataset is None or env.datasets[env.probe_dataset].probe_file is None:
        raise RunError(f"{env.name} has no probe dataset; set EnvSpec.probe_dataset and DatasetSpec.probe_file")

    probe_file = Path(env.datasets[env.probe_dataset].probe_file).expanduser()
    if not probe_file.exists():
        raise RunError(f"probe file not found: {probe_file}")

    params = {**DEFAULT_PROBES, **params}
    probe_set = build_from_exorl(probe_file, params["num"], params["length"], params["seed"], env.factors)
    probe_set.save(path)
    path.with_suffix(".json").write_text(json.dumps({"source": str(probe_file), **params}, indent=2) + "\n")
    return path


def extract(run: ResolvedRun, probe_name: str = "default", environment: dict[str, str] | None = None) -> dict[str, Any]:
    """Run the algorithm's extraction script on the latest checkpoint and record it in run.json."""
    record_path = run.run_dir / "run.json"
    if not record_path.exists():
        raise RunError(f"{run.run_dir} has no run.json; train it first")

    checkpoint = run.adapter.find_checkpoint(run)
    if checkpoint is None:
        raise RunError(f"no checkpoint found under {run.native_dir}")

    probes = probe_set_path(run.root, run.env, probe_name)
    if not probes.exists():
        raise RunError(f"probe set {probe_name!r} not found at {probes}; build it with `prl probes {run.env.name}`")

    out_dir = run.run_dir / "probes" / probe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        command = run.adapter.extract_command(run, checkpoint, probes, out_dir)
    except ValueError as error:
        raise RunError(str(error)) from error
    argv = run.adapter.spec.runtime.argv(command, run.root)

    with open(out_dir / "extract.log", "wb") as log:
        process = subprocess.run(argv, cwd=command.cwd, env={**os.environ, **command.env, **(environment or {})},
                                 stdout=log, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        raise RunError(f"extraction failed (exit {process.returncode}); see {out_dir / 'extract.log'}")

    probe_set = ProbeSet.load(probes)
    np.savez(out_dir / "metadata.npz", **probe_set.metadata)
    info = json.loads((out_dir / "extract_info.json").read_text())

    record = json.loads(record_path.read_text())
    record.setdefault("extractions", {})[probe_name] = {
        "at": _now(),
        "probe_set": str(probes),
        "checkpoint": str(checkpoint),
        "options": run.config.extract,
        "info": {key: value for key, value in info.items() if key not in ("checkpoint", "representations")},
        "representations": info["representations"],
        "dir": str(out_dir),
    }
    _write_record(run, record)
    return record["extractions"][probe_name]


# Added to every analysis as a reference point: what the raw probe state already gives a linear readout.
BASELINE_REPRESENTATION = "observation (baseline)"


def analyze(run: ResolvedRun, probe_name: str = "default", analyzer_names: list[str] | None = None) -> dict[str, Any]:
    """Run analysers on a run's extracted representations; outputs go to analysis/<probe set>/<analyser>/."""
    from predictive_representations_rl.analysis import ANALYZERS
    from predictive_representations_rl.analysis.base import load_extraction

    probe_dir = run.run_dir / "probes" / probe_name
    if not (probe_dir / "metadata.npz").exists():
        raise RunError(f"no extraction for probe set {probe_name!r} in {run.run_dir}; run `prl extract` first")

    names = analyzer_names or list(ANALYZERS)
    unknown = set(names) - set(ANALYZERS)
    if unknown:
        raise RunError(f"unknown analyses {sorted(unknown)}; available: {sorted(ANALYZERS)}")

    representations, metadata = load_extraction(probe_dir)
    representations[BASELINE_REPRESENTATION] = ProbeSet.load(probe_set_path(run.root, run.env, probe_name)).final_observations

    results = {}
    for name in names:
        output_dir = run.run_dir / "analysis" / probe_name / name
        ANALYZERS[name].run(representations, metadata, output_dir)
        results[name] = str(output_dir)

    record_path = run.run_dir / "run.json"
    record = json.loads(record_path.read_text())
    record.setdefault("analyses", {}).setdefault(probe_name, {}).update({name: {"at": _now(), "dir": out} for name, out in results.items()})
    _write_record(run, record)
    return results


def compare_linear_probes(runs: list[ResolvedRun], probe_name: str = "default") -> Path:
    """One table and heatmap of every run's linear-probe scores, rows labelled <algo>[/<task>]/seed_N: <representation>."""
    from predictive_representations_rl.analysis.linear_probe import plot_scores, read_rows, write_rows

    rows, baseline_done = [], False
    for run in runs:
        label = run_label(run)
        for row in read_rows(run.run_dir / "analysis" / probe_name / "linear_probe" / "linear_probe.csv"):
            if row["representation"] == BASELINE_REPRESENTATION:
                if baseline_done:
                    continue
                rows.append(row)
            else:
                rows.append({**row, "representation": f"{label}: {row['representation']}"})
        baseline_done = True

    env = runs[0].env.name
    output_dir = runs_dir(runs[0].root) / env / "analysis" / probe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(rows, output_dir / "linear_probe_comparison.csv")
    plot_scores(rows, output_dir / "linear_probe_comparison.png", f"Linear probe on {env}: all runs")
    return output_dir


def run_label(run: ResolvedRun) -> str:
    return "/".join(filter(None, [run.config.algo, run.config.task, f"seed_{run.seed}"]))


def compare_cka(runs: list[ResolvedRun], probe_name: str = "default") -> Path:
    """Linear CKA between every representation of every run (and the observation baseline) on the shared probe set."""
    from predictive_representations_rl.analysis.base import load_extraction
    from predictive_representations_rl.analysis.cka import cka_matrix, plot_matrix, write_matrix

    probe_set = ProbeSet.load(probe_set_path(runs[0].root, runs[0].env, probe_name))
    representations = {}
    for run in runs:
        probe_dir = run.run_dir / "probes" / probe_name
        if not (probe_dir / "metadata.npz").exists():
            raise RunError(f"no extraction for probe set {probe_name!r} in {run.run_dir}; run `prl extract` first")
        values, metadata = load_extraction(probe_dir)
        if not np.array_equal(metadata.get("source_index"), probe_set.metadata.get("source_index")):
            raise RunError(f"{run.run_dir} was extracted on a different version of probe set {probe_name!r}; re-extract it")
        representations.update({f"{run_label(run)}: {name}": array for name, array in values.items()})
    representations[BASELINE_REPRESENTATION] = probe_set.final_observations

    names, matrix = cka_matrix(representations)
    output_dir = runs_dir(runs[0].root) / runs[0].env.name / "analysis" / probe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_matrix(names, matrix, output_dir / "cka.csv")
    plot_matrix(names, matrix, output_dir / "cka.png", f"Linear CKA between representations on {runs[0].env.name}")
    return output_dir


def _claim_run_dir(run: ResolvedRun, force: bool) -> None:
    """Make the run directory ours. An earlier run is never reused (algorithms may resume from its checkpoints)."""
    record_path = run.run_dir / "run.json"

    if record_path.exists():
        status = json.loads(record_path.read_text()).get("status")
        if status not in STARTABLE:
            if not force:
                raise RunError(f"{run.run_dir} already has a {status} run; pass --force to move it aside and run again")
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run.run_dir.rename(run.run_dir.with_name(f"{run.run_dir.name}.replaced-{stamp}"))
    elif run.run_dir.exists() and any(run.run_dir.iterdir()):
        raise RunError(f"{run.run_dir} exists without run.json; move or delete it first")

    run.native_dir.mkdir(parents=True, exist_ok=True)


def _initial_record(run: ResolvedRun, argv: list[str], env: dict[str, str], cwd: Path, status: str) -> dict[str, Any]:
    spec = run.adapter.spec
    requirement = spec.modes[run.config.mode]

    data_source: dict[str, Any] = {"kind": requirement}
    if requirement == "dataset":
        data_source.update(name=run.config.dataset, files=list(run.env.datasets[run.config.dataset].files))
    elif requirement == "replay":
        data_source.update(replay_from=run.config.replay_from)

    return {
        "status": status,
        "started_at": _now(),
        "seed": run.seed,
        "config_file": str(run.config_path) if run.config_path else None,
        "config": asdict(run.config),
        "env": {"name": run.env.name, "suite": run.env.suite, "task": run.config.task, "obs_type": run.config.obs_type,
                "native_name": run.native_env_name},
        "algorithm": {"name": spec.name, "mode": run.config.mode, "data_requirement": requirement,
                      "task_specific": spec.task_specific, "representations": list(spec.representations)},
        "data_source": data_source,
        "experience": run.adapter.experience(run),
        "command": {"argv": argv, "cwd": str(cwd), "env": env, "shell": shlex.join(argv)},
        "runtime": repr(spec.runtime),
        "host": socket.gethostname(),
        "lsf_job_id": os.environ.get("LSB_JOBID"),
        "git": _git_commits(run.root),
    }


def _write_record(run: ResolvedRun, record: dict[str, Any]) -> None:
    path = run.run_dir / "run.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str) + "\n")
    tmp.replace(path)


def _git_commits(root: Path) -> dict[str, str | None]:
    """Commit (and dirty flag) of the harness and every submodule, so a run can be reproduced."""
    commits: dict[str, str | None] = {}
    repos = [Path(".")] + sorted(path.relative_to(root) for path in (root / "third_party").iterdir() if (path / ".git").exists())

    for repo in repos:
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root / repo, capture_output=True, text=True, check=True).stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root / repo, capture_output=True, text=True, check=True).stdout.strip()
            commits[str(repo)] = sha + ("-dirty" if dirty else "")
        except (subprocess.CalledProcessError, FileNotFoundError):
            commits[str(repo)] = None

    return commits


def _lsf_script(run: ResolvedRun) -> str:
    resources = {**DEFAULT_RESOURCES, **run.config.resources}
    name = f"{run.config.algo}_{run.config.env}_s{run.seed}"
    prl = Path(sys.executable).parent / "prl"
    lines = [
        "#!/bin/bash",
        f"#BSUB -J {name}",
        f"#BSUB -q {resources['queue']}",
        f"#BSUB -n {resources['cores']}",
        '#BSUB -R "span[hosts=1]"',
        f'#BSUB -R "rusage[mem={resources["mem_gb"]}GB]"',
        f"#BSUB -W {resources['walltime']}",
        f"#BSUB -o {run.run_dir}/lsf_%J.out",
        f"#BSUB -e {run.run_dir}/lsf_%J.out",
    ]
    if resources["gpus"]:
        lines.append(f'#BSUB -gpu "num={resources["gpus"]}:mode=exclusive_process"')
    lines += [
        "",
        f"cd {shlex.quote(str(run.root))}",
        f"{shlex.quote(str(prl))} run {shlex.quote(str(run.config_path))} --seed {run.seed}",
        "",
    ]
    return "\n".join(lines)


def _lsf_environment() -> dict[str, str]:
    """bsub needs LSF's profile, which non-login shells on the DTU cluster do not load."""
    if os.environ.get("LSF_ENVDIR"):
        return dict(os.environ)
    output = subprocess.run(["bash", "-c", "source /lsf/conf/profile.lsf >/dev/null 2>&1; env -0"], capture_output=True)
    return dict(item.split("=", 1) for item in output.stdout.decode().split("\0") if "=" in item)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
