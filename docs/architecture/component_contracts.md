# Component contracts

The `typing.Protocol` interfaces in `src/aeris/contracts/`. They build on the
[domain models](domain_models.md); the [architecture overview](overview.md) shows how components
that satisfy them are wired together by the workflows.

## Purpose and dependency direction

Contracts standardize **behavior at component boundaries**, not algorithm internals. A new agent or a new compatible environment must not require editing the experiment orchestrator.

```text
contracts  ──────→  domain
implementations ─→ contracts + domain
workflows ───────→ implementations + contracts + domain

domain -X→ contracts / torch / gymnasium / FB implementation
```

The contracts use `typing.Protocol` for structural compatibility. The runtime `isinstance()` capability checks used in tests are **shallow**: `@runtime_checkable` checks that methods exist, not their argument/return types, valid action spaces, or correct RL semantics. **mypy (run in CI) is what verifies that a component really satisfies a contract**; it caught, for example, `FourRooms.observation_for` accepting fewer states than `TabularModel` promises. A mock implementation still does not establish scientific correctness.

## One module per component, optional agent capabilities

| Module | Protocol | Responsibility | Not its responsibility |
|---|---|---|---|
| `contracts/agent.py` | `Agent` | `act(ActionRequest) -> ActionResult` | Interacting with environment; changing parameters during action selection |
| | `TrainableAgent[BatchT]` | `update(batch) -> tuple[MetricValue, ...]` | Constructing data from a particular dataset; choosing training cadence |
| | `Checkpointable` | Supply/restore opaque learning state | Picking checkpoint paths or writing files |
| | `TaskConditioner[EvidenceT]` | Infer a policy condition from authorized evidence | Obtaining hidden evaluation rewards; updating weights |
| | `ZeroShotAgent[EvidenceT]` | `Agent` + `TaskConditioner`: what zero-shot evaluation needs | Learning during evaluation |
| `contracts/environment.py` | `Environment` | `reset`, `step`, `close` | Choosing agent actions or computing analysis metrics |
| | `TabularModel` | Expose `spec`, exact `tabular_dynamics()`, `observation_for(state)` and the environment's goal-task format, for oracle analyses only | Being used by learning code |
| | `TransitionTable` | Enumerated states and an integer successor table (implemented by `environments.tabular.TabularDynamics`) | Rewards or task definitions (see `TabularMDP`) |
| `contracts/dataset.py` | `TransitionSource` | Read one-step recorded experience lazily | Choosing training batches or neural-network encodings |
| | `BatchProvider[BatchT]` | Construct next strategy-specific batch | Changing the learner or the raw dataset |
| `contracts/recorder.py` | `RunRecorder` | Record manifest, training metrics, evaluation | Deciding experiment configuration or scientific comparability |
| | `TrajectoryRecorder` | Record diagnostic evaluation episodes | Selecting which episodes to keep |
| | `RunReader` | Return the recorded manifest, metrics, evaluations and trajectories as typed records | Interpreting results or checking analysis eligibility |
| | `ArtifactStore` | Save/load derived policy snapshots and analysis reports | Computing those artifacts |

All these capabilities are grouped by component; there is **no universal do-everything agent**. `TrainableAgent` includes `Agent`, while `TaskConditioner` and `Checkpointable` are independent optional capabilities. No algorithm-name branching exists in these definitions.

## Explicit meanings and invariants

### Agent

- `act(request)` selects one action and optional next recurrent policy state. It cannot change model parameters or interact with an environment; input/output representation validity is an implementation/adapter obligation.
- `deterministic=True` affects action sampling, not an assertion that the policy is optimal. A `condition` is an **already inferred** policy condition, not an external task or an inference evidence pointer.
- `update(batch)` takes the **prepared input** required by that learner. A transition pair, time sequence, or latent-conditioned sample can be a different batch type; the training strategy and `BatchProvider` own this construction. Diagnostics use `MetricValue` with explicit units. The agent never opens a replay file or decides how to record the metrics.
- `Checkpointable` returns an opaque Python mapping that may contain tensors, optimizer state and local RNG state. It is **not** necessarily JSON or safely serializable with arbitrary pickling. The recorder supplies storage policy; the orchestrator separately preserves sampler and environment state as required for reproducible resume.
- `TaskConditioner` receives an external `TaskSpec` and **resolved, authorized evidence**, not a mere `TaskEvidenceRef`. The evaluation protocol must resolve and verify evidence permissions first. Zero-shot inference cannot perform learning updates.

### Environment

- We added two minimal runtime domain objects: `ResetResult` and `StepResult`. They are not a new simulator architecture. An adapter may translate a Gymnasium `reset()`/`step()` tuple to these types without importing Gymnasium into core.
- `StepResult` returns the successor observation, an optional reward, and **known** boolean `terminated`/`truncated`. The runner assembles a `Transition` with the original observation and action. Offline transitions may instead use `None` for **unknown** boundaries.
- Runtime `info` is intentionally opaque and **not automatically persisted**, especially if it contains unrecordable objects or privileged data. An artifact serializer would need an explicit schema.
- Environments and agent configs expose a `SpaceSpec` (flat observations, discrete actions). `SpaceSpec.require_compatible` refuses mismatches, e.g. when a checkpoint is analyzed; there is no resolver yet that checks a whole experiment spec before training.

### Data

- `TransitionSource.iter_transitions()` is a lazy data exchange path; it need not load entire datasets into memory. It doesn't promise that repeated iterations are cheap, identical, or supported for an unbounded stream. Concrete sources must declare these guarantees.
- `BatchProvider.next_batch()` leaves sampling algorithm-specific, allowing independent transition pairs, sequences, or imagined-rollout inputs. The runner/training strategy owns scheduling and sampler checkpoint state.

### Recording

- `RunRecorder` deals in previously defined self-describing records. One new model, `TrainingMetricRecord`, connects metric values to a named step axis so gradient updates and environment interactions cannot be mistaken for each other.
- `data.RunDirectory` is the concrete store. It rewrites the manifest atomically, appends record streams (serializing before writing, so a rejected record never leaves a partial line), reads everything back as typed records with file/line/field errors, and stores policy snapshots and versioned reports (`ARTIFACT_NAME`, `schema_version`). Training refuses a non-empty output directory rather than overwriting. Checkpoint files are owned by the agent package (`agents/fb/checkpoint.py`).

## Minimal contract examples

```python
from aeris.contracts import Agent, TrainableAgent
from aeris.domain import ActionRequest, ActionResult, MetricValue


class RandomPolicy:
    def act(self, request: ActionRequest) -> ActionResult:
        return ActionResult(action=0)  # example only; real action must be valid


# No inheritance needed: RandomPolicy satisfies Agent structurally.
```

Tests verify discrete/simple vector observations, toy reward-free environment steps, time-limit distinctions, different training batch structures, optional inference and checkpoint capabilities, typed storage round trips, and self-describing records.

## Open design questions

1. **Richer spaces and a resolver.** `SpaceSpec` covers flat observations and discrete actions. Continuous actions (needed for one-step FB and TD-JEPA on continuous control) and a resolver that checks a whole experiment spec before training are open.
2. **Task inference evidence.** `experiments.evaluate_zero_shot` infers a condition from supplied evidence through `ZeroShotAgent` (`Agent` + `TaskConditioner`) and records the evidence as a `TaskEvidenceRef` on the `EvaluationRecord`. The Four Rooms workflow builds evidence only from the pretraining dataset (or a privileged goal sample, labelled as such), never from evaluation rollouts. A general, permission-scoped evidence view (demonstrations, held-out reward splits) is still open; the contract alone cannot prevent leakage.
3. **Large artifacts.** Trajectories are JSON Lines, suited to small observations; image observations, videos and representation dumps need a binary artifact format.
4. **Training strategies.** Batch construction and update cadence live in per-experiment training workflows; a reusable strategy abstraction should wait until a second training workflow (online, or another algorithm) shows what is common.
5. **Resume.** Agent and sampler state are checkpointed, but restoring a run mid-training is not implemented.
