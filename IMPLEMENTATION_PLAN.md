# Implementation plan: flat zero-shot RL repository

**Status:** plan v2, 2026-10-20. Replaces the earlier plans for an FB-vs-DreamerV3 harness and the rl_lab
platform (both in git history, tagged `dreamer-comparison-2026-10` by step 0).

## Goal

The thesis studies **what forward-backward-style representations learn from the environment** in zero-shot
reinforcement learning: agents pretrain without rewards on offline data, then infer a task from a few rewards and
act zero-shot. The repository becomes one flat codebase organised around that workflow:

```text
environment + offline dataset ──► agent pretraining (reward-free) ──► zero-shot evaluation per task
                                              │
                                              └──► representation analysis (probes, PCA, CKA)
```

## Decisions

| Topic | Decision |
|---|---|
| Agents | **FB, one-step FB, TD-JEPA** only. HILP, Laplacian, ICVF and BYOL-γ are not carried over. |
| Third-party code | **All submodules removed.** The needed onestep-fb code moves into the repository (MIT licence and upstream commit kept in `agents/LICENSE`). Offline_vs_Online_in_MBRL (DreamerV3) is dropped entirely. |
| Environments | An `environments/` package that supports several environment suites behind one interface (ExORL/DMC and OGBench to start). |
| rl_lab | **Removed**: `packages/rl-lab`, `catalog/`, `implementation_lab.md`. |
| Old contracts | **Removed**: the whole `src/predictive_representations_rl` package (`core/data.py`, `core/backend.py`, `AlgorithmSpec`/`EnvSpec`, compatibility checks, runtimes, adapters, extraction subprocess scripts, export). Pieces worth keeping are rewritten as plain modules. |
| Python | One environment: Python 3.10 + JAX (CUDA 12), as onestep-fb uses. Everything runs in-process. |
| Results | `RESULTS.md` kept; its Dreamer sections marked historical. Existing FB runs in `runs/` are kept as read-only legacy data. |

## Target structure

```text
predictive-representations-rl/
├── agents/                     # zero-shot RL agents
│   ├── __init__.py             # AGENTS = {"fb": ..., "onestep_fb": ..., "td_jepa": ...}; default configs
│   ├── fb.py
│   ├── onestep_fb.py
│   ├── td_jepa.py
│   ├── networks.py             # MLPs, value/actor heads, ensembles (from onestep-fb utils/networks.py)
│   ├── encoders.py             # pixel encoders (from utils/encoders.py)
│   ├── flax_utils.py           # TrainState, ModuleDict, save/restore (from utils/flax_utils.py)
│   └── LICENSE                 # onestep-fb MIT licence + upstream commit
├── environments/               # environment suites behind one interface
│   ├── __init__.py             # make_env(name, task), load_datasets(name), list_tasks(name), task_reward(...)
│   ├── registry.py             # per environment: suite, tasks, dataset files, task-definition files, factors
│   ├── datasets.py             # Dataset / GCDataset sampling, relabelling with task rewards
│   ├── exorl/                  # DMC-based ExORL domains
│   │   ├── env.py              # DMC → gymnasium wrapper, make_env
│   │   ├── data.py             # load ExORL hdf5, relabel rewards
│   │   ├── generate.py         # download + build datasets (from data_gen_scripts/)
│   │   └── tasks/              # custom_dmc_tasks (point_mass_maze, walker, ...)
│   ├── ogbench/                # OGBench (AntMaze, cube, scene, ...)
│   │   ├── env.py
│   │   └── data.py
│   └── factors.py              # ground-truth state factors for analysis (point_mass_maze; AntMaze later)
├── training/
│   ├── pretrain.py             # reward-free pretraining loop (from onestep-fb main.py)
│   ├── zero_shot.py            # task inference (z from rewards) + evaluation per task
│   ├── run.py                  # one experiment: config → run directory (run.json, logs, checkpoints)
│   └── logging.py              # CSV/W&B logging (from utils/log_utils.py)
├── analysis/
│   ├── probe_sets.py           # fixed probe states from held-out data, with factors
│   ├── representations.py      # named views per agent (FB: backward, forward, q_values, latent; TD-JEPA: its encoders), untrained controls
│   ├── linear_probes.py, pca.py, cka.py
│   └── plots.py
├── configs/<environment>/<agent>.yaml   # sweeps as small files using `extends`
├── scripts/
│   ├── submit_lsf.py           # submit a config/seed to the DTU cluster (exclusive GPU, shared mode)
│   ├── status.py               # list runs and their LSF state
│   └── generate_datasets.sh
├── tests/
├── runs/                       # outputs (git-ignored)
├── main.py                     # CLI: train | evaluate | extract | analyze | submit | status
├── RESULTS.md  IMPLEMENTATION_PLAN.md  README.md
└── pyproject.toml              # Python 3.10, one project
```

### Where things come from

| New location | From |
|---|---|
| `agents/fb.py`, `onestep_fb.py`, `td_jepa.py` | `third_party/onestep-fb/agents/` |
| `agents/networks.py`, `encoders.py`, `flax_utils.py` | `third_party/onestep-fb/utils/` |
| `environments/exorl/`, `ogbench/`, `datasets.py` | `third_party/onestep-fb/envs/`, `utils/{env_utils,datasets}.py`, `data_gen_scripts/` |
| `environments/registry.py`, `factors.py` | `src/.../envs/` (rewritten without the EnvSpec contract) |
| `training/pretrain.py`, `zero_shot.py`, `logging.py` | `third_party/onestep-fb/main.py`, `utils/{evaluation,log_utils}.py` |
| `training/run.py`, `scripts/submit_lsf.py`, `status.py` | `src/.../runner.py` (run.json, LSF, status), simplified |
| `analysis/` | `src/.../probes/`, `src/.../extract/onestep_fb_extract.py`, `packages/rl-lab/src/rl_lab/analysis/`, `src/.../analysis/plotting.py` |
| config `variant` / `extends` | `src/.../core/config.py` |

## Removed

- Submodules `third_party/onestep-fb` (after vendoring, step 3) and `third_party/Offline_vs_Online_in_MBRL`;
  `.gitmodules`; `third_party/onestep-fb/.venv`.
- DreamerV3: `.venvs/dreamerv3` (12 GB), `runs/point_mass_maze/dreamerv3/` (1.4 GB), Dreamer configs and scripts.
- Agents HILP, Laplacian, ICVF, BYOL-γ.
- `src/predictive_representations_rl/` (old contracts and harness), `packages/rl-lab/`, `catalog/`,
  `implementation_lab.md`, `scripts/smoke_test_fb_dataset.py`, the old tests.

## Steps

Each step ends tested and committed.

**0. Safety.** Tag the current commit `dreamer-comparison-2026-10`. Note the existing FB runs to keep.

**1. Remove DreamerV3 and rl_lab.** Delete the Dreamer submodule, venv, runs, configs and scripts; delete
`packages/rl-lab`, `catalog/`, `implementation_lab.md`. Mark the Dreamer sections of `RESULTS.md` historical.
Frees ~13.5 GB.

**2. New skeleton and environment.** Create the flat directories and a Python 3.10 `pyproject.toml` with the
onestep-fb dependencies (JAX CUDA 12, Flax, Optax, dm_control, gymnasium, OGBench, h5py) plus analysis
dependencies (scikit-learn, matplotlib, PyYAML).
*Check:* `uv sync`; JAX uses the GPU on a GPU node and the CPU on the login node.

**3. Move the agent, environment and training code in, while the submodule still exists.** FB, one-step FB,
TD-JEPA and their shared modules; ExORL and OGBench environments; the pretraining and zero-shot loop. Rewrite
imports for the new layout.
*Check (equivalence):* on CPU with the same seed and config, a 2,000-step run of `training/pretrain.py` and of the
submodule's `main.py` give identical `train.csv`/`eval.csv` and identical checkpoint parameters, for one-step FB
on point_mass_maze. Then remove the onestep-fb submodule.

**4. Run management.** `training/run.py` (config → run directory with `run.json`: status, commit, config,
task-definition fingerprint, timing), `main.py` CLI, `scripts/submit_lsf.py` (exclusive GPU in shared mode, runs
directory passed explicitly), `scripts/status.py` (marks vanished jobs lost), configs with `variant`/`extends`.
*Check:* a short CPU run end to end; dry-run LSF script generation.

**5. Analysis.** Probe sets, representation extraction for the three agents (with untrained controls), linear
probes, PCA, CKA and plots as plain modules. Remove `src/predictive_representations_rl/`.
*Check:* re-analysing the existing FB runs' extracted representations reproduces the numbers in `RESULTS.md`
exactly; TD-JEPA extraction runs on a short CPU checkpoint.

**6. Environments beyond point_mass_maze.** Registry entries for the other ExORL domains and OGBench AntMaze;
AntMaze factors (x, y, maze cell, distance to goals) and probe set.
*Check:* probe sets build for point_mass_maze and AntMaze; factors unit-tested.

**7. Baselines.** One-step FB on AntMaze large-navigate with the README settings reaches the published zero-shot
range (validates the moved code); FB, one-step FB and TD-JEPA on point_mass_maze with the new structure.
*Check:* results added to `RESULTS.md`.

**8. Documentation.** README: structure, install, quick start (generate data → train → evaluate → extract →
analyze), cluster usage.

## Risks

| Risk | Handling |
|---|---|
| Moved code silently differs from onestep-fb | Step 3 equivalence check on CPU before the submodule goes; upstream commit recorded |
| Shared modules pull in code of removed agents | Only modules used by FB, one-step FB and TD-JEPA are moved; unused code paths deleted; tests cover all three agents |
| Generic top-level package names (`agents`, `environments`, `training`, `analysis`) | Imported only within this project; renamed if a clash appears |
| GPU non-determinism | Bitwise checks only on CPU (step 3) and for analysis (step 5) |
| Losing earlier work | Everything recoverable from the tag; findings kept in `RESULTS.md` |

## Open questions (do not block the restructure)

1. FB settings for point_mass_maze (the walker settings give no working zero-shot policy): sweep reward
   temperature and discount after the AntMaze reproduction.
2. Seeds per configuration for thesis results (suggested: 3).
3. Whether the existing legacy runs need to be readable by the new analysis code, or only their recorded results.
