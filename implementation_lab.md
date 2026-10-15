# RL Representation Lab — Implementation Specification

**Status:** v0.2 (2026-10-15), revision of v0.1 after the first FB vs DreamerV3 results  
**Primary interface:** Streamlit  
**Primary language:** Python  
**Goal:** One place to gather, inspect and compare what reinforcement-learning agents represent: their latent
representations, how those relate to the environment's state, and what the agent does, with the controls and
comparability checks that make such comparisons defensible.

> **Instruction to the implementation agent:** Build what the thesis needs first, in the phase order of §11, on
> real exported runs. Put generality into the data contract and the plugin interfaces, not into speculative
> features. Never couple scientific analyses to Streamlit, and never branch on algorithm or environment names in
> the core. Keep exactly one implementation of each analysis. If a required input is missing, say why an analysis
> cannot run instead of fabricating a result.

### What changed from v0.1

| Area | v0.1 | v0.2 |
|---|---|---|
| Purpose | generic platform | thesis-first, contribution-ready (§1) |
| Relation to existing code | built from scratch | `rl_lab` owns the analysis code; the training harness (`prl`) exports bundles and calls `rl_lab` (§3) |
| Controls | "baselines" for probes only | first-class control representations, included in every comparison (§5.4) |
| Seeds | deferred | experiment groups and seeds in the data model; aggregation in Phase 1 (§5.2, §7) |
| Comparability | sample-ID alignment only | training data, objective, task-definition fingerprint and extraction protocol recorded and checked (§6) |
| Diagnostic dataset | a table the user supplies | shared, versioned artifact with a documented construction protocol (§5.3) |
| Stochastic latents | not covered | extraction protocol records sampling vs mode and a noise estimate (§5.5) |
| Rollouts / video | trajectory table + video | rollout extraction: per-step states, actions, rewards, frames **and** latents along the agent's own episodes (§5.6) |
| Storage | copy into catalog, SQLite | reference in place, filesystem catalog; SQLite deferred (§9) |
| First milestone | synthetic fixture | real point_mass_maze runs; synthetic fixture for tests and demos (§11) |
| Deferred | — | schema migrations, hardened ZIP import, multi-user, SQLite (§14) |

---

## 1. Purpose and scientific priorities

The thesis asks **what different RL algorithms learn from the environment**. Results come from many runs
(algorithms, settings, seeds, checkpoints) and many analyses. The lab must make them easy to browse and compare
in one place, and later explain behaviour state by state and step by step.

It should also become a **contribution of its own**: a documented format and explorer that others can use to
compare what RL agents represent. That value lies in three things generic tools (TensorBoard, W&B, embedding
projectors) do not provide:

1. **A shared diagnostic format**: the same probe states, with stable IDs, evaluated by different algorithms, with
   each representation's extraction protocol recorded.
2. **Safe defaults**: controls, leakage-safe splits, seed aggregation and comparability warnings built in.
3. **State-level linking**: one state inspected across all representations, values, neighbours, trajectories and
   video frames.

### Scientific rules (from experience with the first runs)

- **A score without its control is uninterpretable.** Untrained networks already decode position at R² ≈ 0.96;
  results are shown next to untrained and raw-observation controls by default.
- **One seed is an anecdote.** Single-seed probe scores swung by up to ±0.4 between runs; aggregates over seeds
  are the default view whenever a group has several seeds, and single-seed results are labelled as such.
- **Compare like with like, or say so.** Runs trained on different data, objectives or reward definitions can be
  compared, but the differences are shown next to the result (§6).
- Distinguish **what is decodable** (probes), **how it is structured** (PCA, CKA, neighbourhoods), **what
  correlates with behaviour** (values, rollouts) and **what transfers**. None of these alone shows causal use.

### Non-goals for v0.x

- Training agents or running checkpoints from the web interface.
- Loading untrusted model code or serialized checkpoints (`.pkl`, `.pt`, JAX checkpoints) in the lab.
- Public multi-user hosting.
- Reimplementing experiment trackers, job schedulers or general embedding viewers.
- Interpreting unlabelled arrays automatically: artifact roles come from the manifest.

---

## 2. Architectural rules

1. **Data-first, algorithm-agnostic.** The lab consumes a versioned bundle format. Algorithm-specific exporters
   live outside the lab core.
2. **Data ≠ analysis ≠ presentation.** Bundle reading, scientific computation, result caching and Streamlit
   rendering are separate modules. Analysis code never imports Streamlit.
3. **One analysis implementation.** PCA, probes, CKA etc. exist once, in `rl_lab.analysis`, and are used by both
   the lab and the training harness.
4. **Capability-driven.** Analyses declare requirements and are checked against the data actually present.
5. **Controls and comparability are part of the core,** not a UI afterthought.
6. **Reproducible.** Results record analysis version, input content hashes, parameters, seeds and splits.
7. **Offline-first, headless-capable.** Everything except rendering runs from the CLI on the cluster.
8. **Graceful degradation.** Metrics only → curves; embeddings → structure analyses; labels → probes;
   rollouts → trajectory views; video → playback.

### Pipeline

```text
Native training (onestep-fb, DreamerV3, future)      ← run by the training harness (prl)
        │
        ▼
Extraction: probe-set representations, controls, rollouts (prl extract / prl rollout)
        │
        ▼
Exporter (prl export, algorithm-specific) ──writes──▶ bundle (versioned format, §5)
        │
        ▼
rl_lab: index catalog → validate → capabilities/comparability → analyses (cached) → Streamlit explorer
```

---

## 3. Packages and repository layout

`rl_lab` is a separate, installable package inside this repository (a `uv` workspace member), with **no
dependency** on the training harness, JAX or the third-party repos. It can move to its own repository later
without changes. The harness (`predictive_representations_rl`) depends on `rl_lab` for analysis and for the bundle
writer.

```text
predictive-representations-rl/
├── packages/rl-lab/                    # the lab: own pyproject, publishable
│   ├── pyproject.toml                  # extras: [ui] for streamlit/plotly, [video] for imageio-ffmpeg
│   ├── src/rl_lab/
│   │   ├── contracts/                  # Pydantic models: manifest, artifacts, datasets, results
│   │   ├── bundle/                     # read + validate bundles; writer helpers used by exporters
│   │   ├── catalog/                    # filesystem catalog, index, content hashes, result cache
│   │   ├── analysis/
│   │   │   ├── base.py                 # plugin contract, eligibility, results
│   │   │   ├── registry.py
│   │   │   ├── comparability.py        # §6
│   │   │   └── plugins/                # pca, effective_rank, linear_probe, linear_cka, knn, ...
│   │   ├── ui/                         # Streamlit app, pages, shared components
│   │   └── cli.py                      # rl-lab index | validate | analyses | analyze | app | fixture
│   └── tests/                          # incl. tiny synthetic bundles
├── src/predictive_representations_rl/  # the training harness (prl)
│   ├── analysis/                       # → moved into rl_lab.analysis (thin wrappers remain)
│   ├── export.py                       # prl export: runs/ → bundles
│   └── ...
└── ...
```

**Dependency direction:** `contracts` → `bundle`/`catalog` → `analysis` → `ui`. Exporters depend only on
`rl_lab.contracts` and `rl_lab.bundle`.

### Stack

| Concern | Choice | Note |
|---|---|---|
| UI | Streamlit + Plotly | `rl_lab[ui]` extra, so headless use needs neither |
| Numerics | NumPy, SciPy, scikit-learn | existing analysis code moves here |
| Tables | pandas + PyArrow (Parquet) | samples, metrics, rollout steps |
| Arrays | `.npz` (`allow_pickle=False`) | Zarr later if arrays outgrow memory |
| Schemas | Pydantic + YAML | versioned manifest |
| Catalog | filesystem + JSON index | SQLite only if listing/querying becomes slow |
| Video | MP4 (H.264) via `imageio-ffmpeg` on the export side | lab only plays files |
| Tests | pytest | |

No FastAPI, Redis, Celery, React or remote storage in v0.x.

---

## 4. Concepts

| Concept | Meaning |
|---|---|
| **Experiment** | one trained run: algorithm × environment × configuration × seed |
| **Group** | experiments that differ only by seed (e.g. `onestep_fb/point_mass_maze/alpha0`) |
| **Checkpoint** | a saved training step of an experiment |
| **Diagnostic dataset** | a fixed, shared set of input states with stable `sample_id`s and ground-truth labels (§5.3) |
| **View** | one named representation of an experiment at a checkpoint on a dataset, e.g. `B`, `F`, `rssm_deter` |
| **Control view** | a reference view for interpretation: untrained network, raw observation, random projection (§5.4) |
| **Rollout** | episodes produced by running the trained agent, with per-step records, latents and frames (§5.6) |

---

## 5. Bundle format (schema 1.0)

A **bundle** is one experiment. **Diagnostic datasets** are separate, shared artifacts that bundles reference by
ID and content hash, so many experiments can be compared on the same samples without duplicating them.

### 5.1 Layout

```text
catalog/
├── datasets/
│   └── point_mass_maze-probes-default/
│       ├── dataset.yaml                     # construction protocol, source, hash (§5.3)
│       ├── samples.parquet                  # sample_id, episode_id, timestep, labels
│       └── inputs.npz                       # optional: observations/actions per sample (for observation controls)
└── experiments/
    └── onestep_fb--point_mass_maze--alpha0--s0/
        ├── manifest.yaml
        ├── representations/
        │   └── step_1000000/
        │       ├── B.npz                    # sample_ids + embeddings
        │       ├── B__untrained.npz
        │       └── ...
        ├── metrics/
        │   ├── training.parquet             # training_step, metric, value
        │   └── evaluation.parquet           # training_step, task, metric, value
        ├── rollouts/                        # optional (§5.6)
        └── provenance/                      # copied run.json, config, logs (small files only)
```

Bundles may **reference** large files in place (e.g. under `runs/`) instead of copying them; the manifest records
the path and content hash, and validation fails clearly if the file changed or disappeared.

### 5.2 Manifest

```yaml
schema_version: "1.0"
exporter: {name: "prl-export", version: "0.1.0", source_revision: "a272340"}
experiment:
  id: "onestep_fb--point_mass_maze--alpha0--s0"
  group: "onestep_fb/point_mass_maze/alpha0"
  seed: 0
  algorithm: {name: "one-step-fb", revision: "e2d29de", configuration: {latent_dim: 50, alpha: 0, discount: 0.98}}
  environment:
    id: "dmc/point_mass_maze"
    task_definition:                     # what the reward/termination actually are (§6)
      source: "onestep-fb envs/custom_dmc_tasks/point_mass_maze.py"
      fingerprint: "sha256:…"            # hash of the task code (+ XML)
    tasks: [reach_top_left, reach_top_right, reach_bottom_left, reach_bottom_right]
  training:
    mode: "offline"                       # offline | online
    objective: "reward_free"              # reward_free | task_reward
    trained_task: null                    # e.g. reach_top_left for task_reward
    data: {kind: "dataset", id: "exorl-rnd-point_mass_maze", transitions: 5_000_000}
    budget: {gradient_steps: 1_000_000, env_steps: 0}
views:
  - id: "B"
    role: "trained"                       # trained | control
    checkpoint_step: 1_000_000
    dataset: {id: "point_mass_maze-probes-default", hash: "sha256:…"}
    path: "representations/step_1000000/B.npz"
    protocol:                             # how the view was computed (§5.5)
      input: "final_state_and_action"     # or: window_filter
      window_length: 16
      conditioning: {type: "none"}        # FB F(s,a), z-conditioned views state their z source
      stochastic: false
  - id: "B__untrained"
    role: "control"
    control_kind: "untrained"             # untrained | observation | random_projection
    control_of: "B"
    checkpoint_step: 0
    dataset: {id: "point_mass_maze-probes-default", hash: "sha256:…"}
    path: "representations/step_1000000/B__untrained.npz"
    protocol: {input: "final_state_and_action", window_length: 16, conditioning: {type: "none"}, stochastic: false}
artifacts:
  - {kind: "metrics", path: "metrics/training.parquet"}
  - {kind: "metrics", path: "metrics/evaluation.parquet"}
```

Unknown files are ignored; roles come only from the manifest. `schema_version` major mismatches are rejected with
a clear message; migrations are deferred until a second schema version exists.

### 5.3 Diagnostic datasets

The diagnostic dataset is the foundation of every comparison, so its construction is documented, not assumed:

```yaml
id: "point_mass_maze-probes-default"
hash: "sha256:…"
source: {kind: "held_out_dataset", id: "exorl-rnd-point_mass_maze-val", note: "never used for training"}
construction: {method: "random_windows", num_samples: 4096, window_length: 16, seed: 0,
               constraints: ["within one episode", "excludes the placeholder final step"]}
labels:
  x: {kind: continuous, unit: "m"}
  room: {kind: categorical, classes: 4}
  rewarded_reach_top_left: {kind: binary, definition: "reward > 1e-6 under task_definition sha256:…"}
grouping: {column: "episode_id"}          # used for leakage-safe splits
coverage: {room_counts: {0: 1364, 1: 1967, 2: 499, 3: 266}}
```

`samples.parquet` holds one row per sample: `sample_id` (stable string), `episode_id`, `timestep`, label columns.
Splits are not stored as a fixed column by default; analyses build grouped folds from `grouping` and record them.

### 5.4 Controls

Every comparison should be readable against references. Control views are ordinary views with
`role: control`:

- **untrained**: the same network at initialisation (same architecture, config and seed), extracted with the same
  protocol. Exporters should always provide it when the training code allows it.
- **observation**: the raw input from `inputs.npz`, computed by the lab itself.
- **random_projection**: a fixed random linear map of the observation to the trained view's width, computed by
  the lab (controls for representation width).

Analyses show controls next to the views they control by default (probe tables, CKA matrices, PCA). The UI can
hide them, but never hides that they exist.

### 5.5 Extraction protocol

Representations depend on how they are computed; the protocol makes that explicit and comparable:

- `input`: `final_state_and_action` (FB) or `window_filter` (recurrent models filtered over the sample's window).
- `window_length`, initial-state convention, action alignment (`a_{t-1}` feeds step `t`).
- `conditioning`: for task-conditioned views (FB `z`, Q-values), the task and how `z` was inferred.
- `stochastic`: whether sampled latents were used; if so `posterior: sample | mode` and, when measured,
  `sampling_noise` (run-to-run difference relative to across-sample spread).

Views with different protocols can still be compared, with a warning (§6).

### 5.6 Rollouts (Phase 3)

To see what an agent does, the exporter runs the trained agent and records its own episodes:

```text
rollouts/<rollout_id>/
├── rollout.yaml          # policy (e.g. FB z for task X; Dreamer task policy), episodes, seeds, deterministic?
├── steps.parquet         # episode_id, timestep, sample_id, action_*, reward, done, label columns, frame_index
├── representations/      # per-step views keyed by sample_id (same view ids as the probe set)
└── video/<episode_id>.mp4  # one frame per timestep: frame_index == timestep (exact alignment)
```

Rollout samples form their own dataset (`rollout:<rollout_id>`); they can be projected into a probe-set analysis
(e.g. the probe-set PCA basis) for "where is the agent in representation space right now?".

### Data conventions

- Embeddings: `sample_ids` (1-D string array, unique) and `embeddings` (`(N, D)`, finite float32/64).
- Training metrics: `training_step`, `metric`, `value`, optional `wall_time_s`. Evaluation metrics add `task`.
  Training steps are never confused with rollout timesteps.
- Units of steps are explicit (`agent_steps`, `env_steps`, `gradient_steps`); Dreamer's action repeat makes these
  differ.

---

## 6. Capabilities and comparability

### Capabilities

A `CapabilityResolver` answers derived queries with reasons, e.g. `has_views(dataset)`, `has_labels(name)`,
`has_controls(view)`, `has_matched_views(a, b)`, `has_checkpoints(view, ≥2)`, `has_group_seeds(group, ≥2)`,
`has_rollouts`, `has_video_mapping(rollout)`. An analysis is **available** or **unavailable (reasons)**; execution
state is `not_run | running | completed | failed` and is tracked separately.

### Comparability report

For any selection of experiments/views, the lab computes a report shown next to every comparison:

| Check | If different |
|---|---|
| diagnostic dataset (id + hash) | **block** probe/CKA comparisons across them |
| task-definition fingerprint | **warn** (e.g. dense vs sparse reward under one environment name) |
| training data (kind/id), objective, trained task | **show** (expected in cross-algorithm work, but must be visible) |
| budget / experience | **show** |
| extraction protocol (input, window, stochastic/posterior) | **warn** |
| view width | **show**; offer the random-projection control and PCA-k-matched probes |
| seeds per group | **warn** for n = 1 |

---

## 7. Analyses

### Plugin contract

```python
@dataclass(frozen=True)
class Eligibility:
    available: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

class AnalysisPlugin(Protocol):
    id: str            # "linear_probe"
    version: str       # bumps invalidate cached results
    scope: str         # "view" | "views" (cross-view) | "group" | "rollout"
    def check(self, catalog, request) -> Eligibility: ...
    def compute(self, context, request) -> AnalysisResult: ...
```

Plugins are registered explicitly in a registry; nothing from a bundle is executed. Results are cached under
`catalog/results/<analysis>/<cache-key>/` (`result.json` with inputs, hashes, parameters, versions, warnings;
tables as Parquet; figures optional). The cache key covers analysis id + version, input content hashes, view,
dataset, labels, parameters, splits and seeds.

### Analyses by phase

| Analysis | Scope | Phase | Notes |
|---|---|---|---|
| PCA + spatial maps | view | 1 | maps when position labels exist; shared reference basis for cross-checkpoint plots |
| Effective rank / spectrum | view | 1 | participation ratio, components for 90% |
| Linear probe | view | 1 | ridge R² / balanced accuracy; grouped CV (stratified for classes); rare classes skipped with reasons; train-only standardisation |
| Linear CKA | views | 1 | requires identical sample IDs; read against controls |
| Seed aggregation | group | 1 | mean ± spread for all of the above |
| Width-matched probe | view | 2 | probe on top-k PCs, k equal across views |
| Non-linear (MLP) probe | view | 2 | tells "reorganised" from "discarded" |
| k-NN neighbourhood agreement | views | 2 | per-sample overlap of nearest neighbours between views, mappable onto the environment |
| Checkpoint evolution | view × checkpoints | 2 | needs intermediate checkpoints from the exporter |
| Rollout projection | rollout | 3 | per-step latents in the probe-set basis |
| Value/feature along trajectories | rollout | 3 | e.g. Q per task, distance to goal, per step |
| Topology / reachability, RSA / Procrustes | — | later | |

### Rules

- Probes: join by `sample_id`; folds grouped by episode; standardise on training folds only; report chance and
  control scores; never report a probe without its control when one exists.
- CKA: identical sample IDs in identical order; estimator stated (centred linear); upward bias for wide views
  noted; controls in the same matrix.
- PCA across checkpoints/experiments: one shared reference basis, or explicitly separate fits.
- Uncertainty: single-seed results are labelled; group views show spread.
- No analysis implies causal use of a feature.

---

## 8. Streamlit experience

### Phase 1 pages

1. **Library**: experiments grouped by environment → algorithm → group → seed; status, checkpoints, views,
   available analyses; filter and multi-select.
2. **Experiment**: manifest summary, training/evaluation curves, views with their protocols, analysis cards
   (available / unavailable with reasons / results).
3. **Compare**: select experiments/groups; probe table with controls inline and seed aggregation; CKA matrix;
   effective-rank table; PCA maps side by side; the comparability report always visible above the results.

### Phase 2: state explorer

Select a sample (on the environment map, in a projection, or from a table) and see: its labels and position; its
coordinates in each view's projection; its nearest neighbours in each view (shown on the map); per-task values
where available; and maps of per-sample metrics (e.g. neighbourhood agreement between FB and Dreamer).

### Phase 3: trajectories and video

Choose a rollout and episode: video (`st.video`) plus a timestep slider as the authoritative position; the
synchronised panel shows action, reward, labels, values and the step's latent in the shared projection, with the
trajectory traced through representation space. No promise of frame-accurate player callbacks.

### Usability rules

- Every result shows its parameters, seeds, controls and warnings; downloads for tables and figures.
- Expensive analyses run via `rl-lab analyze` (cluster-friendly); the UI reads cached results and only runs
  cheap analyses inline.
- Interactive scatters cap points with a reproducible sampling seed and say so.

### Access

Run `rl-lab app --catalog <path>` on the cluster and open it through an SSH tunnel
(`ssh -L 8501:localhost:8501 <login-node>`), or sync the catalog (small: probe-set views are a few MB per
experiment) to a laptop and run it there.

---

## 9. Catalog, storage and safety

- The catalog is a directory (`datasets/`, `experiments/`, `results/`, `index.json`). `rl-lab index` scans it and
  validates bundles; adding an experiment = exporting a bundle into it (or registering a path).
- Large files may be referenced in place with content hashes; the home-directory quota on the cluster makes
  copying checkpoints or rollouts expensive.
- Safety for local use: folder import only; `allow_pickle=False`; reject paths escaping the bundle and symlinks;
  accepted formats YAML, JSON, Parquet, NPZ, MP4. Hardened ZIP upload (size/compression limits) comes with any
  hosted use.

---

## 10. Exporters (outside the lab core)

`prl export <config> [--seed]` turns a harness run into a bundle: it writes views from `probes/<set>/` (trained
and `_untrained`), registers the probe set as a dataset, converts native logs (FB `train.csv`/`eval.csv`, Dreamer
`metrics.jsonl`) to metrics Parquet, fills `training`/`task_definition`/`protocol` from `run.json` and the adapter,
and copies small provenance files.

Producer-side work the lab depends on (harness, not lab):

| Need | Harness change | Phase |
|---|---|---|
| Task-definition fingerprint | hash task code + XML at run time, store in `run.json` | 1 |
| Seeds | run groups with `seeds: [0, 1, 2]` | 1–2 |
| Checkpoint evolution | keep intermediate checkpoints (FB `save_interval`; copy Dreamer's periodic checkpoint) — disk cost | 2 |
| Rollouts | `prl rollout`: run the trained agent per task, record steps, frames and per-step latents | 3 |

Another research group adds an algorithm by writing an exporter against `rl_lab.contracts`; no lab changes.

---

## 11. Phases and acceptance

### Phase 1 — Results browser (thesis-critical)

- `rl_lab` package skeleton (workspace member), contracts, bundle reader/validator, writer helpers.
- Move PCA, effective rank, linear probe and CKA from the harness into `rl_lab.analysis` as plugins; the harness
  calls them (one implementation; existing tests move with them).
- Capability resolver and comparability report.
- `prl export`; task-definition fingerprint recorded by new runs (and computed for existing ones from the repo).
- Streamlit Library, Experiment and Compare pages.
- Synthetic fixture bundle for tests.

**Acceptance:** the six existing point_mass_maze runs are exported and browsable; the Compare page reproduces the
probe, CKA and dimensionality tables in `RESULTS.md` (same numbers), with controls and the comparability report
showing "different data / objective / trained task"; analyses on unchanged inputs are not recomputed after a
restart; unavailable analyses state why.

### Phase 2 — State explorer and depth

- State explorer page; k-NN neighbourhood agreement with per-sample maps.
- Width-matched and non-linear probes; seed aggregation with ≥ 2 seeds per group.
- Checkpoint evolution (requires intermediate checkpoints from the harness).

**Acceptance:** selecting a state shows it in every view with neighbours on the map; group results show mean ±
spread; at least one experiment shows a checkpoint evolution with a shared PCA basis.

### Phase 3 — Rollouts and video

- `prl rollout` (harness) for FB (per task) and Dreamer (its task).
- Rollout pages: video + timestep slider + synchronised latent/value/action panel; trajectories in the probe-set
  basis.

**Acceptance:** for one FB and one Dreamer rollout, moving the slider shows the matching frame, action, reward and
latent; missing mappings are explained.

### Phase 4 — Contribution hardening

- Documentation (format spec, exporter guide, plugin guide), README quick start with `rl-lab fixture create`.
- Packaging (`pip install rl-lab`), CI tests, hardened ZIP import.
- Second environment (see §13) to show environment independence.

**Acceptance:** a new user can install the lab, create the demo fixture and explore it in minutes without any
training code; adding a plugin or an exporter needs no core changes.

---

## 12. Tests

- **Contracts:** valid/invalid manifests and datasets; missing files; unsupported schema; NaNs; duplicate or
  mismatched sample IDs; changed referenced-file hashes.
- **Analyses:** existing numerical tests move with the code (PCA invariances, probe signal vs noise, grouped CV,
  CKA rotation/scale invariance); controls computed by the lab (observation, random projection) are correct.
- **Comparability:** a dataset-hash mismatch blocks CKA; a task-fingerprint mismatch warns; n = 1 groups warn.
- **Caching:** the key changes with data, parameters, analysis version, dataset or split.
- **End to end:** fixture → index → analyses → results persisted → UI smoke test (Streamlit AppTest); export of a
  real harness run round-trips.
- **Safety:** path traversal, symlinks, pickled NPZ rejected.

---

## 13. Open decisions

1. **Environment for the next stage.** point_mass_maze: both algorithms run, but FB's zero-shot policy fails with
   the walker settings. OGBench AntMaze: FB has published settings and works, but the DreamerV3 fork has no
   OGBench support (needs an environment adapter). The choice drives which runs Phase 2–3 are built on.
2. **Seeds per group** for the thesis comparisons (e.g. 3), given ~1.5 h (FB) and ~4.5 h (Dreamer) per run.
3. **Which checkpoints** to keep for evolution analyses, given 250 MB per Dreamer checkpoint and the home quota.
4. **Reward-free Dreamer** (world model without reward head) as an additional group, to separate method from
   objective.
5. **Name and licence** for the published package.

## 14. Deferred, not forgotten

Schema migrations (until a v2 schema exists), SQLite catalog, hardened ZIP upload, multi-user hosting, background
job queue / HPC workers, frame-accurate custom video component, UMAP, topology/reachability, RSA/Procrustes,
interventions, launching training from the lab.

**Guiding question:** *What was actually provided, and what scientifically defensible comparison can we show
from it, with its controls and caveats in view?*
