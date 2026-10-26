"""``aeris sweep``: train and analyze one experiment for several seeds."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer

from aeris.cli._common import cli_errors

app = typer.Typer(
    help="Run an experiment for several seeds (locally or on LSF).", no_args_is_help=True
)


@app.command(
    "four-rooms",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def four_rooms(
    ctx: typer.Context,
    name: Annotated[str, typer.Option(help="Experiment name shared by the seeds.")],
    seeds: Annotated[str, typer.Option(help="Seeds, e.g. 0-4 or 0,3,7.")] = "0-4",
    root: Annotated[Path, typer.Option("--root", "-r", help="Runs directory.")] = Path("runs"),
    lsf: Annotated[
        bool, typer.Option("--lsf", help="Submit an LSF job array (one job per seed).")
    ] = False,
    local: Annotated[
        bool, typer.Option("--local", help="Run the seeds one after another here.")
    ] = False,
    queue: Annotated[str, typer.Option(help="LSF queue.")] = "hpc",
    walltime: Annotated[str, typer.Option(help="LSF wall-clock limit per seed (H:MM).")] = "4:00",
    memory: Annotated[str, typer.Option(help="LSF memory per seed.")] = "4GB",
) -> None:
    """Train and analyze NAME for every seed. Extra options go to `aeris train four-rooms`.

    Without --lsf or --local, only prints what would run. Example:

        aeris sweep four-rooms --name fb_oracle --seeds 0-4 --lsf --updates 40000
    """
    from aeris.workflows.sweeps import SeedSweep, parse_seeds, sweep_summary

    with cli_errors():
        if lsf and local:
            raise ValueError("choose either --lsf or --local")
        sweep = SeedSweep(
            name=name, seeds=parse_seeds(seeds), train_args=tuple(ctx.args), root=root
        )
        existing = [str(sweep.run_dir(s)) for s in sweep.seeds if sweep.run_dir(s).exists()]
        if existing:
            raise ValueError(f"run directories already exist: {', '.join(existing)}")
        for line in sweep_summary(sweep):
            typer.echo(line)
        if lsf:
            script = sweep.lsf_script(queue=queue, walltime=walltime, memory=memory)
            path = Path("logs/sweeps") / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.lsf"
            typer.echo(sweep.submit_lsf(script, script_path=path))
            typer.echo(f"Job script: {path}  (follow with: bjobs; logs in logs/sweeps/)")
        elif local:
            failed = {seed: code for seed, code in sweep.run_locally().items() if code != 0}
            if failed:
                raise RuntimeError(f"seeds failed (exit codes): {failed}")
            typer.echo("All seeds finished.")
        else:
            typer.echo("\nDry run. Each seed runs:")
            for command in sweep.commands(sweep.seeds[0]):
                typer.echo("  aeris " + " ".join(command[3:]))  # drop "<python> -m aeris.cli.main"
            typer.echo("Add --lsf to submit a job array, or --local to run here.")
