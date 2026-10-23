# Component Contracts — Step 3

**Status:** Tested first pass, **not frozen**. Depends on [Experiment Lifecycle](experiment_lifecycle.md) and [Domain Models](domain_models.md).

## Purpose and dependency direction

Contracts standardize **behavior at component boundaries**, not algorithm internals. A new agent or a new compatible environment must not require editing the experiment orchestrator.

```text
contracts  ──────→  domain
implementations ─→ contracts + domain
orchestrator ────→ contracts + domain

domain -X→ contracts / torch / gymnasium / FB implementation
```

The contracts use `typing.Protocol` for structural compatibility. The runtime `isinstance()` capability checks used in tests are **shallow**: `@runtime_checkable` checks that methods exist; it does not confirm their argument/return types, valid action spaces, or correct RL semantics. Static type-checking and boundary tests remain necessary. Avoid assuming that a mock implementation establishes scientific correctness.

## One module per component, optional agent capabilities

| Module | Protocol | Responsibility | Not its responsibility |
|---|---|---|---|
| `contracts/agent.py` | `Agent` | `act(ActionRequest) -> ActionResult` | Interacting with environment; changing parameters during action selection |
| | `TrainableAgent[BatchT]` | `update(batch) -> tuple[MetricValue, ...]` | Constructing data from a particular dataset; choosing training cadence |
| | `Checkpointable` | Supply/restore opaque learning state | Picking checkpoint paths or writing files |
| | `TaskConditioner[EvidenceT]` | Infer a policy condition from authorized evidence | Obtaining hidden evaluation rewards; updating weights |
| `contracts/environment.py` | `Environment` | `reset`, `step`, `close` | Choosing agent actions or computing analysis metrics |
| `contracts/dataset.py` | `TransitionSource` | Read one-step recorded experience lazily | Choosing training batches or neural-network encodings |
| | `BatchProvider[BatchT]` | Construct next strategy-specific batch | Changing the learner or the raw dataset |
| `contracts/recorder.py` | `RunRecorder` | Record manifest, training metrics, evaluation | Deciding experiment configuration or scientific comparability |

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
- Observation/action-space descriptions and compatibility validation remain open. We **must implement explicit space descriptors and a resolver check before training real FB**. A shared function signature alone is insufficient.

### Data

- `TransitionSource.iter_transitions()` is a lazy data exchange path; it need not load entire datasets into memory. It doesn't promise that repeated iterations are cheap, identical, or supported for an unbounded stream. Concrete sources must declare these guarantees.
- `BatchProvider.next_batch()` leaves sampling algorithm-specific, allowing independent transition pairs, sequences, or imagined-rollout inputs. The runner/training strategy owns scheduling and sampler checkpoint state.

### Recording

- `RunRecorder` deals in previously defined self-describing records. One new model, `TrainingMetricRecord`, connects metric values to a named step axis so gradient updates and environment interactions cannot be mistaken for each other.
- A concrete recorder will need to define its idempotence/overwrite/transaction rules, run status, failure handling, artifact and checkpoint storage, and schema versioning. Do **not** claim full experiment reproducibility or analysis readiness based on these three methods alone.

## Minimal contract examples

```python
from thesis.contracts import Agent, TrainableAgent
from thesis.domain import ActionRequest, ActionResult, MetricValue

class RandomPolicy:
    def act(self, request: ActionRequest) -> ActionResult:
        return ActionResult(action=0)  # example only; real action must be valid

# No inheritance needed: RandomPolicy satisfies Agent structurally.
```

Tests verify discrete/simple vector observations, toy reward-free environment steps, time-limit distinctions, different training batch structures, optional inference and checkpoint capabilities, and self-describing records. They deliberately **do not** yet define a configured experiment runner.

## Deferred design choices (intentionally unresolved)

1. **Compatibility/SpaceSpec:** Determine a backend-independent way to describe observation/action modalities, shapes, and bounds, then check them at resolution time, before costly training. First mock runner should exercise one compatible and one incompatible combination.
2. **Task inference evidence:** A resolved evidence view (reward-labeled samples, demonstrations, or something else) must be permission-scoped, with explicit provenance; `TaskConditioner` alone cannot prevent evaluation leakage.
3. **Recorder artifacts:** Add indexed trajectories/representations/videos and full error and checkpoint metadata only with a real storage implementation.
4. **Training workflow:** How to select and instantiate `BatchProvider` and update cadence lives in a training strategy, not `Agent` or `Environment`.
5. **Typing/serialization:** Future static checks, device transfers, batch shapes, process boundaries, and distributed execution require additional validation, not an ever-wider base protocol.

## Step 3 acceptance

- [x] Existing Step 2 domain tests still pass.
- [x] Protocols are importable without Gymnasium, PyTorch, or FB.
- [x] Fake policy and trainable agent satisfy common interface without inheriting a base class.
- [x] Different algorithm batch shapes share a generic data source and training boundary.
- [x] External evidence inference is an optional capability, not mandatory for agents.
- [x] Environment steps distinguish termination and truncation.
- [x] Metric recording identifies task/checkpoint for evaluation and axis for training.
- [ ] One configured runner swaps environments and agents without source edits (Step 4).
- [ ] Compatibility, leakage and resume behavior are verified end-to-end (Step 4 and subsequent steps).
- [ ] Any FB result is reproduced or scientifically validated (later steps).
