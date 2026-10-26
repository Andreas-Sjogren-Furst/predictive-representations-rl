# Seeds, sweeps and summaries

A single run is one sample. AERIS compares configurations by training several **seeds** of each,
then reporting **mean ± standard deviation across seeds**.

## Naming

`aeris train four-rooms --name NAME --seed K` records `NAME` as the experiment name and
`NAME_seedK` as the run id, and writes to `runs/NAME_seedK` unless `--output` is given. All seeds of
one configuration share `NAME`, which is how they are grouped later. The seed drives the agent's
initialization, batch sampling, evaluation start states and, for `--dataset random`, the collected
data itself.

## Running a sweep

```bash
aeris sweep four-rooms --name fb_oracle --seeds 0-4 --updates 40000                 # dry run: prints the plan
aeris sweep four-rooms --name fb_oracle --seeds 0-4 --lsf --updates 40000           # LSF job array
aeris sweep four-rooms --name fb_random --seeds 0-4 --lsf --dataset random --updates 40000
aeris sweep four-rooms --name smoke --seeds 0-2 --local --updates 100               # here, one after another
```

Every option `aeris sweep` does not know (`--updates`, `--dataset`, `--hidden-dim`, …) is passed to
`aeris train four-rooms`. Each seed trains and then runs its analyses: the tabular policy analysis,
the zero-shot task suite, representations, and, for oracle data, the FB occupancy diagnostic.

With `--lsf`, one job per seed is submitted as an LSF job array (`--queue hpc`, `--walltime 4:00`,
`--memory 4GB` by default; each job uses one core, as torch runs single-threaded). The generated job
script is kept in `logs/sweeps/` and every job logs to `logs/sweeps/NAME_<job>_<index>.out`; follow
the jobs with `bjobs`. The script spells out every command with absolute paths, because DTU's LSF does
not pass the submitting shell's environment to jobs. A sweep refuses to start if any of its run
directories already exists.

A 40 000-update Four Rooms run with the default networks takes about 1.5 hours on one core.

## Summaries

```bash
aeris runs summary                                   # group by experiment name
aeris runs summary --group-by dataset.origin -m zero-shot
aeris runs summary --json > summary.json
```

Groups are formed by any recorded value: `name`, `seed`, or `<component>.<key>` with component
`agent`, `dataset`, `environment`, `training` or `evaluation` (e.g. `agent.hidden_dim`,
`training.updates`). Only completed runs count. For each group it prints the mean ± sample standard
deviation and the number of seeds of every headline number: final evaluation, final policy
optimality and all-start success, the FB occupancy diagnostic, and the zero-shot suite score per
family and evidence protocol.

In the dashboard, the multi-run panels (training curves, suite scores) have a **Group runs by**
option that offers the experiment name and every setting that differs between the selected runs:
curves become mean ± std bands, bars get ± std error bars, and the *Group summary* panel shows the
same table as `aeris runs summary`. The *Compare runs* layout uses them.
