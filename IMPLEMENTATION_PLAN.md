# Lightweight RL Experiment Harness

## Goal

Build a **thin experimental harness** that makes it easy to run existing RL algorithms on selectable environments, collect their outputs in one standard format, and analyse their learned representations.

```text
Environment ⟂ Algorithm ⟂ Analysis
```

Principles:

- **Run algorithms as they are.** The harness does not replace or rewrite Dreamer, FB, or any other implementation, and does not force them into a common training mode. Each algorithm runs in its native mode (online, offline, ...) with its native replay buffer or dataset loader.
- **Describe, don't constrain.** Each algorithm declares what it is (online/offline, data it needs, action/observation types, representations it exposes). The harness uses this to pick the right launch path and to reject incompatible env × algorithm combinations before a run starts.
- **Record, don't enforce.** Experience and compute (transitions seen, gradient steps, data source) are recorded for every run so they can be compared afterwards; they are not forced to be equal at this stage.
- **One small adapter per algorithm or environment.** Adding one must not require changes elsewhere.
- **Analysis only reads standard artifacts**, never algorithm-specific log directories.

---

## 1. Architecture: an orchestrator, not a shared process

The algorithms cannot share a Python process with each other or with the harness:

| Component | Python | Key dependencies | Runtime |
|---|---|---|---|
| Harness (`predictive_representations_rl`) | 3.12 | numpy (+ analysis libs) | `uv` project venv |
| `third_party/onestep-fb` | 3.10 | JAX/Flax, gymnasium 0.29 | `third_party/onestep-fb/.venv` |
| `third_party/Offline_vs_Online_in_MBRL` (DreamerV3) | 3.9 | JAX + TensorFlow, gym 0.19 | Singularity/Docker image or own venv |

So the harness **orchestrates subprocesses**. Every run has three stages:

```text
            harness (py3.12)                         algorithm runtime (own venv/container)
┌───────────────────────────────────┐        ┌──────────────────────────────────────────┐
│ 1. resolve config + check compat  │──cmd──►│ train: the repo's own entry point         │
│                                   │        │   (main.py / dreamerv3/train.py)          │
│ 2. extract                        │──cmd──►│ extract: small script in harness/extract/ │
│                                   │◄─.npz──│   loads checkpoint, encodes probe set     │
│ 3. analyse (in-process)           │        └──────────────────────────────────────────┘
└───────────────────────────────────┘
```

Extraction scripts live in this repo (not in the submodules) but are executed with the algorithm's interpreter, so they can import the algorithm's code. They exchange data with the harness only through `.npz`/`.json` files.

---

## 2. Algorithm descriptors and adapters

### Descriptor: what the algorithm is

```python
@dataclass(frozen=True)
class AlgorithmSpec:
    name: str                                          # "dreamerv3", "onestep_fb"
    description: str
    modes: Mapping[LearningMode, DataRequirement]      # {"online": "self_collected", "offline": "replay"}
    action_spaces: frozenset[str]                      # {"continuous"} / {"discrete", "continuous"}
    observation_types: frozenset[str]                  # {"state", "pixels"}
    representations: tuple[str, ...]                  # names exposed by extraction
    stateful_representation: bool                      # True if the representation depends on history (RNN)
    task_specific: bool                                # True: trains on one task's reward; False: task-agnostic pretraining
    runtime: Runtime                                   # Venv(path) | Container(image)
```

`DataRequirement` is what a mode needs before it starts: `self_collected` (interacts and fills its own replay), `dataset` (a fixed offline dataset from the env registry), or `replay` (another run's replay directory).

Initial descriptors:

| | `dreamerv3` | `onestep_fb` |
|---|---|---|
| modes → data | online → self_collected; offline (passive) → replay | offline → dataset (ExORL hdf5 / OGBench npz) |
| training | single task | task-agnostic pretraining, zero-shot on every task |
| actions | discrete, continuous | continuous only |
| observations | state (`dmc_proprio`), pixels (`dmc_vision`) | state, pixels |
| representations | `deter` (h), `stoch` (z), `model_state` (h,z) | `backward` B(s), `forward` F(s,a,z), `latent` z per task |
| stateful | yes | no |

### Adapter: how to run it

```python
class AlgorithmAdapter(ABC):
    spec: AlgorithmSpec

    @abstractmethod
    def train_command(self, run: ResolvedRun) -> Command:
        """argv + env vars that launch the repo's own training entry point."""

    @abstractmethod
    def find_checkpoint(self, run_dir: Path) -> Path:
        """Locate the checkpoint the repo wrote."""

    @abstractmethod
    def extract_command(self, run: ResolvedRun, probe_set: Path, out_dir: Path) -> Command:
        """Command that writes <representation>.npz for the probe set."""

    def collect_metrics(self, run_dir: Path) -> Iterator[dict]:
        """Translate the repo's native logs into standard metrics.jsonl rows."""
```

The adapter only builds command lines, maps names, and parses outputs. All learning happens in the original code.

### Compatibility check

Before launching, the runner checks `env.action_type ∈ algo.action_spaces`, `obs_type ∈ algo.observation_types`, `mode ∈ algo.learning_modes`, and that the required data source exists (e.g. the ExORL hdf5 file). Failures produce a clear message such as *"onestep_fb does not support discrete actions"* instead of a crash mid-run.

---

## 3. Environment registry

Environments are selected by name. Each entry describes the environment once and maps it to each algorithm's own naming:

```python
@dataclass(frozen=True)
class EnvSpec:
    name: str                              # "point_mass_maze"
    suite: str                             # "dmc"
    tasks: tuple[str, ...]                 # ("reach_top_left", ...)
    action_type: str                       # "continuous"
    observation_types: frozenset[str]      # {"state", "pixels"}
    datasets: dict[str, DatasetSpec]       # {"exorl_rnd": DatasetSpec(...)}
    algo_names: dict[str, Callable]        # algo -> (task, obs_type) -> native env/task string
    factors: Callable | None               # ground-truth factors for analysis
```

Example native names for `point_mass_maze`, task `reach_top_left`:

- one-step FB: `--env_name=exorl-rnd-point_mass_maze` (trains on all tasks, evaluates zero-shot on each)
- DreamerV3: `--task dmc_point_mass_maze_reach_top_left --configs dmc_proprio`

For controlled environments, `factors()` exposes interpretable state variables (for point_mass_maze: `x`, `y`, `vx`, `vy`, distance to each goal corner).

---

## 4. Data sources and replay buffers

- **Algorithms keep their own replay buffers.** Dreamer's online run fills its own replay directory; FB loads its own dataset. The harness does not reimplement replay.
- The harness manages **offline data sources** only: where a dataset lives, whether it exists, how to obtain it (download/generate commands), and its size.
- Where data must move between algorithms (e.g. Dreamer replay → FB dataset, or ExORL → Dreamer passive replay) the existing canonical types in `core/data.py` (`Trajectory`, `RepresentationDataset`, `TaskDataset`) are the intermediate format, with one converter per algorithm format (`backends/fb.py::representation_dataset_to_fb` is the first). Not needed for milestone 1.

---

## 5. Experiment configs and CLI

One small YAML file per experiment; anything under `overrides` is passed through to the algorithm unchanged.

```yaml
# configs/point_mass_maze/onestep_fb.yaml
env: point_mass_maze
algo: onestep_fb
mode: offline
dataset: exorl_rnd
obs_type: state
seeds: [0]
budget: {train_steps: 1_000_000}
overrides:
  agent.latent_dim: 50
  agent.discount: 0.98
```

```bash
uv run prl check  configs/point_mass_maze/onestep_fb.yaml            # compatibility + data check only
uv run prl run    configs/point_mass_maze/onestep_fb.yaml --seed 0   # train → extract → metrics
uv run prl sweep  configs/point_mass_maze/*.yaml --seeds 0-2         # many runs
uv run prl run    ... --submit lsf                                   # submit to the cluster instead of running locally
uv run prl analyze runs/point_mass_maze --analysis pca,linear_probe
uv run prl list   envs|algos|runs
```

---

## 6. Standard run directory

```text
runs/                                   # or $PRL_RUNS_DIR (e.g. scratch space)
└── <environment>/
    └── <algorithm>/
        └── <mode>/[<task>/]seed_<N>/   # <task> only for single-task algorithms (Dreamer)
            ├── run.json            # resolved config + provenance (below)
            ├── train.log           # stdout/stderr of the algorithm process
            ├── job.lsf             # when submitted with --submit lsf
            ├── native/             # the repo's own logs/checkpoints, untouched
            ├── metrics.jsonl       # standardized metrics (train + eval)        [not yet]
            ├── probes/<probe set>/
            │   ├── <representation>.npz   # key `values`, [N, D] (FB `latent` is [tasks, D])
            │   ├── metadata.npz           # copy of the probe set's per-probe factors
            │   ├── extract_info.json
            │   └── extract.log
            └── analysis/                                                        [not yet]
```

A run directory is never reused: `--force` moves an existing run aside to `seed_<N>.replaced-<time>` (algorithms such as Dreamer would otherwise silently resume from the old checkpoint).

`run.json` records: status/exit code/duration, algorithm, mode, data requirement, env, task, obs type, data source (dataset files or replay dir), **experience** as configured (env steps, dataset, gradient steps), seed, the exact command line and environment, runtime, host, LSF job id, the checkpoint found after training, and the git commit (+dirty flag) of the harness and of each submodule. This is what makes later fairness comparisons possible without enforcing them now.

---

## 7. Probe sets and representation extraction

Representation analysis uses a fixed set of inputs so all algorithms are compared on exactly the same states.

```python
@dataclass(frozen=True)
class ProbeSet:
    observations: np.ndarray          # [N, T, obs_dim]  short sequences ending at the probe state
    actions: np.ndarray               # [N, T, act_dim]  actions along those sequences
    metadata: dict[str, np.ndarray]   # factors at the final state: x, y, goal distances, ...
```

- **Stateless representations** (FB `B(s)`, `F(s,a,z)`) use only the final state of each sequence.
- **Stateful representations** (Dreamer `h`, `z`) are computed by running the RSSM posterior over the whole sequence; the value at the final step is the representation. A single observation would give a meaningless `h`, so probes are sequences (default `T = 16`) cut from held-out trajectories.
- Probe sets are generated once per environment by the harness and saved under `probes/<env>/<name>.npz`, then passed to every algorithm's extraction script.

---

## 8. Analysis

Analysers receive only arrays and metadata:

```python
class Analyzer(ABC):
    name: str

    @abstractmethod
    def run(self, representations: dict[str, np.ndarray], metadata: dict[str, np.ndarray], output_dir: Path) -> dict:
        ...
```

Initial: PCA, linear probes (predict factors from representation). Later: UMAP, latent-dimension heatmaps, pairwise latent-distance vs. true distance, CKA between checkpoints/algorithms, Procrustes/RSA.

Avoid comparing individual latent coordinates directly; representations may be rotated or permuted while carrying the same information.

---

## 9. Package layout

Builds on the existing package rather than replacing it:

```text
src/predictive_representations_rl/
├── core/
│   ├── data.py            # existing: Trajectory, RepresentationDataset, TaskDataset
│   ├── evaluation.py      # existing: evaluate()
│   ├── backend.py         # existing in-process interface; superseded by algorithms/base.py
│   ├── config.py          # YAML → ResolvedRun
│   ├── runtime.py         # Venv / Container, subprocess launching
│   └── registry.py        # env + algorithm registries
├── envs/
│   ├── base.py            # EnvSpec, DatasetSpec
│   └── dmc.py             # point_mass_maze (+ walker, cheetah, ... later)
├── algorithms/
│   ├── base.py            # AlgorithmSpec, AlgorithmAdapter
│   ├── onestep_fb.py
│   └── dreamerv3.py
├── backends/
│   └── fb.py              # existing: canonical data → FB format
├── extract/               # run inside each algorithm's runtime
│   ├── onestep_fb_extract.py
│   └── dreamerv3_extract.py
├── probes/
│   └── probe_set.py
├── analysis/
│   ├── pca.py
│   └── linear_probe.py
├── runner.py              # check → train → extract → metrics
└── cli.py                 # `prl` entry point
configs/                   # experiment YAML files
runs/                      # outputs (git-ignored)
```

---

## 10. Milestone 1: point_mass_maze, both algorithms, end to end

`point_mass_maze` is a 2D point mass in a maze with four goal corners (`reach_top_left`, `reach_top_right`, `reach_bottom_left`, `reach_bottom_right`). Both repos ship the same ExORL `custom_dmc_tasks` implementation, actions are continuous, and the (x, y) position gives a direct ground truth for the representation analysis.

### 10.1 Make each algorithm runnable on point_mass_maze by hand

**One-step FB (offline, ExORL RND data).** Two bugs in the fork's *environment/dataset plumbing* (not the algorithm) currently break this domain:

1. `envs/exorl_utils.py::make_env_and_datasets`: for a domain-only name, the default task is `'walk'` (or `'reach_bottom_left'` for jaco); `'walk'` is not a point_mass_maze task, so the `ALL_TASKS` assertion fails. Fix: keep `'walk'` where the domain has it, otherwise default to `ALL_TASKS[domain_name][0]` (unchanged for walker/cheetah/quadruped/jaco).
2. `data_gen_scripts/exorl_dataset_aggregator.py::_worker_fn`: task keys are derived with `'_'.join(task.split('_')[1:])`, which turns `point_mass_maze_reach_top_left` into `mass_maze_reach_top_left`. The rewards are then stored as `rewards_mass_maze_...`, but `relabel_dataset` looks up `rewards_reach_top_left`. Fix: pass bare task names to the workers and build the env name from the known domain.

Status: both fixed in the submodule working tree and verified on a small synthetic point_mass_maze buffer (aggregate → hdf5 → `make_env_and_datasets` → `relabel_dataset` for all four tasks). Still to do: commit in the fork (`Andreas-Sjogren-Furst/onestep-fb`), push, and bump the submodule. Then:

```bash
sh data_gen_scripts/exorl_download.sh point_mass_maze rnd
python data_gen_scripts/generate_exorl_dataset.py --domain_name=point_mass_maze --save_path=~/.exorl/data/rnd-point_mass_maze.hdf5
python data_gen_scripts/generate_exorl_dataset.py --domain_name=point_mass_maze --save_path=~/.exorl/data/rnd-point_mass_maze-val.hdf5 --skip_size=5_000_000 --dataset_size=100_000
python main.py --env_name=exorl-rnd-point_mass_maze --agent=agents/onestep_fb.py --train_steps=<small> ...
```

**DreamerV3 (online, proprio).** `dmc_point_mass_maze_reach_bottom_right` is already in the repo's DMC task list and `embodied/envs/dmc.py` routes maze tasks to `custom_dmc_tasks`. Open item: set up a runtime on the HPC (Singularity image from the repo, or a Python 3.9 venv following the Dockerfile) and confirm a short run with `--configs dmc_proprio --task dmc_point_mass_maze_reach_top_left`.

### 10.2 Harness

1. ✅ `AlgorithmSpec`, `EnvSpec`, registries, YAML configs, compatibility check (`prl check`, `prl list`).
2. ✅ `Runtime` (venv, container) and `runner.py`: `prl run <config> [--seed N] [--dry-run] [--submit lsf] [--force]` launches training and writes `run.json`, `train.log`, `native/`.
3. ✅ Adapters: `onestep_fb` and `dreamerv3` online (train command, checkpoint location, configured experience). Still to do: metrics translation to `metrics.jsonl`; Dreamer offline (passive) mode.
4. ✅ Probe sets: `prl probes <env>` cuts 4096 windows of 16 steps from the held-out RND val split into `runs/<env>/probes/default.npz` (metadata: x, y, vx, vy, speed, room/quadrant, distance to each goal, per-task reward, episode/step).
5. ✅ Extraction: `prl extract <config> [--cpu]` runs `extract/<algo>_extract.py` inside the algorithm's runtime on the latest checkpoint and records it under `extractions` in `run.json`.
   - one-step FB: `backward` B(s,a), `forward` F(s,a) (ensemble mean), `latent` z per task (inferred exactly as main.py does), `q_values` F·z per task. B and F take the dataset action at the probe state (clipped as in training).
   - DreamerV3: `deter` (h), `stoch` (z, flattened one-hot), `model_state` (h,z) at the final step of the posterior filtered over the window (prev action a_{t-1}, learned initial state at t=0). Matches the repo's own `forward_obs_pass` within sampling noise. State encoders only.
6. ✅ Analysis: `prl analyze <configs...> [--analysis pca,linear_probe]` writes `analysis/<probe set>/<analyser>/` per run, and with several runs a combined `runs/<env>/analysis/<probe set>/linear_probe_comparison.{csv,png}`. The raw probe observation is always included as a baseline representation.
   - PCA: explained variance (components for 90/95%, participation ratio), and maps of PC1-3 over (x, y).
   - Linear probes: ridge R² for continuous factors; class-balanced logistic regression scored by balanced accuracy (chance 1/k) for categorical ones (room, and `rewarded_<task>` = reward > 1e-6, since rewards are mostly 0). 5-fold CV split by episode (stratified for classes), since overlapping windows from one episode would leak. Classes with < 20 probes or from < 5 episodes are skipped and listed in the summary.
   - Untrained control: both extraction scripts also write every representation from the network at initialisation (same architecture, config and seed) as `<name>_untrained`, so it appears next to the trained one in every analysis.

**Done when:** `prl run` works for both configs on point_mass_maze, each run directory contains `run.json`, `metrics.jsonl` and probe representations, and `prl analyze` produces PCA plots and linear-probe scores for FB `B`/`F` and Dreamer `h`/`z` on the same probe set.

---

## 11. Known issues / open questions

- **Dreamer runtime**: Python 3.9 venv at `.venvs/dreamerv3`, built by `scripts/setup_dreamerv3_venv.sh` (DMC-only subset of the repo's Dockerfile; JAX 0.4.30, TF 2.16). Verified on CPU up to training + checkpoint; a full run still needs a GPU node.
- **No GPU on login nodes**: both JAX installs probe CUDA at start-up; for CPU runs set `environment: {JAX_PLATFORMS: cpu}` in the config.
- **FB is continuous-action only.** Discrete environments (e.g. a gridworld Four Rooms) need either a continuous variant or an FB change; deferred.
- **Dreamer offline (passive) mode is not purely offline**: it steps a training env to pace updates and prefills its eval replay with a random policy. Fine for "run as is"; must be accounted for (and recorded in `run.json`) when comparing experience later.
- **Different experience by default**: FB trains on ~5M RND transitions; Dreamer online collects its own. Recorded, not equalized, in milestone 1.
- **Dreamer representations are stochastic**: z is sampled from the posterior and h depends on those samples. On the tiny debug model the run-to-run noise in h was ~60% of its spread across probes. Check on the trained model; if still large, add a deterministic option (posterior mode / logits) or average several samples.
- **Dreamer action alignment in extraction** is consistent with the repo's own pass but could not be discriminated on the debug model (it barely uses actions); re-check on a trained checkpoint.
- **Linear decodability of the state is not evidence of learning**: the untrained FB networks recover position as well as trained ones (R² ≈ 0.99 on the smoke run). Compare every representation with its `_untrained` control, and lean on analyses that test structure beyond the state (goal distances, rewards, PCA geometry, later CKA/RSA).
- `core/backend.py` (in-process `PredictiveRLBackend`) does not fit the subprocess design; keep until the adapters replace it, then remove.

## 12. Later

- More environments (walker, cheetah, quadruped, jaco; OGBench), and a continuous Four Rooms.
- More analysers (UMAP, CKA, RSA, distance analysis).
- Experience-matched comparisons (shared datasets via the canonical data format).
- Cluster sweeps with result aggregation.

The framework should stay **thin**. If an abstraction requires large changes to the original RL algorithms, it is probably too low-level.
