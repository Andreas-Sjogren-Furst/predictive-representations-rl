# Domain models

The shared vocabulary of AERIS, in `src/aeris/domain/`. It imports nothing else from AERIS and
neither NumPy nor torch (enforced by `tests/test_architecture.py`). See the
[architecture overview](overview.md) for where these records flow and the
[experiment lifecycle](experiment_lifecycle.md) for the requirements they come from.

## Purpose

Standardize **the meaning of data exchanged by components**, without binding the thesis research framework to FB, a particular environment, Gymnasium, Torch, or a storage backend.

This iteration distinguishes two categories:

1. **Declarative and recorded data** — portable, JSON-compatible specs and records. Nested config mappings are copied/frozen at construction to prevent accidental mutation of the research configuration. `to_dict()` and `from_dict()` (see below) convert them to and from JSON data.
2. **Runtime data** — `ActionRequest`, `ActionResult`, `Transition`; their observations/actions/conditions remain opaque because agents and environments may exchange discrete values, float vectors, images, dictionaries, or recurrent memory. These objects are **not** assumed to be JSON serializable.

## Models

| Module | Objects | Design rule |
|---|---|---|
| `domain/experiment.py` | `ExperimentSpec`, `ComponentSpec`, `TrainingPlan`, `EvaluationPlan`, `EvaluationTarget` | Describes choices and strategies, never constructs components |
| `domain/agent.py` | `ActionRequest`, `ActionResult` | Agent task condition is already inferred; recurrent memory is distinct from environment observation |
| `domain/experience.py` | `Transition` | Do not conflate `terminated`, `truncated`, and missing flags; reward may be missing in reward-free data |
| `domain/task.py` | `TaskSpec`, `TaskEvidenceRef` | Task identity, permitted inference evidence, and model-specific condition are different things |
| `domain/results.py` | `RunManifest`, `EvaluationRecord`, `MetricValue` | Results identify run, checkpoint, task, protocol, seed, evidence and metric semantics |
| `domain/trajectory.py` | `EvaluationTrajectory`, `TrajectoryStep` | Observational episode traces; never task-inference evidence |
| `domain/spaces.py` | `SpaceSpec`, `ObservationSpec`, `DiscreteActionSpec` | Components declare what they emit/expect; compatibility is checked by comparing specs, not ad-hoc dimensions |
| `domain/artifacts.py` | `PolicySnapshot` | Derived artifacts are typed; states are matched by identity, never by index order |
| `domain/reports.py` | `AnalysisReport`, `TabularPolicyReport`, `PolicySummary` | Analyses return typed reports; each report names its artifact file. Algorithm-specific reports live with the algorithm (e.g. `agents/fb/reports.py`) |

### Reading records back

Records that are saved as JSON (`RunManifest`, `ExperimentSpec`, `EvaluationRecord`,
`TrainingMetricRecord`, `EvaluationTrajectory`, `SpaceSpec`, `PolicySnapshot`, reports) share the
`JsonRecord` mixin: `to_dict()` writes plain JSON data and `from_dict()` rebuilds the typed record.
`from_dict()` checks the data against the type hints, rejects unknown or missing fields with the
offending path (for example `RunManifest.experiment.seed`), and then runs the record's own
validation. Consumers therefore receive valid objects and do not re-validate raw dictionaries.

### ExperimentSpec

- A **component reference** contains an identifier and optional declarative settings (`{"latent_dim": 64}`). It is not an instantiated class, file path, or import string.
- `TrainingPlan` uses `offline`, `online`, `hybrid`, or `none`. Trainable modes identify a strategy selector; how it samples or schedules updates is intentionally left to the selected strategy.
- `environment` and `dataset` are optional independently, but a training mode may require one or both. An evaluation-only run (`mode="none"`) needs an `initial_checkpoint`.
- `EvaluationPlan` ties a named evaluation protocol to external tasks and *optional* references to inference evidence. A task does not contain a pretrained FB vector `z`. The same task can be evaluated with different inference evidence references, to support coverage/sensitivity experiments.
- `seed` is part of the spec. The actual run gets a separate `run_id` and provenance.
- **Compatibility is not proved by this dataclass.** Observation/action support is checked with `SpaceSpec`; dataset fingerprints, checkpoint/config agreement and run eligibility are checked by the workflows. There is no resolver yet that instantiates components from a spec: the spec is recorded, not executed.

### Runtime model limitations (deliberate)

- `Any` on observations and actions is an exchange point, **not** a claim that all networks support all shapes. `SpaceSpec` describes flat observation vectors and discrete actions so far; richer modalities will need richer specs.
- A single `Transition` models a one-step observation. Sequential training inputs, independent contrastive samples, sequence masks and latent samples are algorithm-specific constructs handled later by training strategies.
- A `None` termination flag means **unknown**. Neither a `False` default nor a time-limit-to-terminal conversion is permissible.
- Extra trajectory provenance, time-varying discounts, multi-agent environments and observation serialization formats are deferred until an actual use case needs them.

### Results and scientific interpretation

- `RunManifest` is a container for a complete configuration snapshot and provenance. The training workflow records the full agent configuration, the dataset fingerprint, library versions, the AERIS git commit with a `git_dirty` flag (both `null` outside a git checkout; see `workflows/provenance.py`) and a `running`/`completed`/`failed` status; hardware is not recorded yet. Constructing an empty-provenance manifest is not certification of reproducibility.
- `EvaluationRecord` represents the result of one task, evaluated at one checkpoint using a named protocol and seed. The `inference_evidence` field contains a typed source/split reference for permitted evidence provenance; it must not be filled from secret evaluation rollouts.
- `MetricValue` explicitly records its unit and aggregation (e.g. `fraction` and `mean`). Equal metric names alone are **not** sufficient for cross-environment comparisons; protocol, task semantics, normalization and environment identity also matter.
- Training metrics, evaluations, trajectories, policy snapshots and reports are stored by `data.RunDirectory` (layout in the [overview](overview.md#life-of-a-run)). Videos, representation dumps and checkpoint resume are not implemented.

## Declarative example

```python
from aeris.domain import ComponentSpec, ExperimentSpec, TrainingPlan

spec = ExperimentSpec(
    name="fb_four_rooms",
    agent=ComponentSpec("fb", {"latent_dim": 32}),
    dataset=ComponentSpec("four_rooms_offline_v1"),
    environment=ComponentSpec("four_rooms"),
    training=TrainingPlan("offline", ComponentSpec("offline_updates", {"updates": 5000})),
    seed=42,
)

config = spec.to_dict()  # pure JSON-compatible snapshot, no algorithm instantiated
assert ExperimentSpec.from_dict(config) == spec
```

See `_experiment_spec` in [`workflows/four_rooms_training.py`](../../src/aeris/workflows/four_rooms_training.py) for a complete experiment specification.

## Boundaries

Contracts and every other package can import these domain classes; **domain classes never import
contracts or implementations**. Agent-specific training inputs and inspection representations remain
optional capabilities rather than mandated fields in `Transition` or `ExperimentSpec`.
