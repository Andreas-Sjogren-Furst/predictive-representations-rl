# First end-to-end discrete FB diagnostic

This integration demonstrates the **existing AERIS contracts** working with a
real discrete FB learner and a four-room grid-world environment.

## Run

```bash
uv sync --extra fb-torch
uv run --extra fb-torch pytest -q
uv run --extra fb-torch python examples/fb_four_rooms.py --updates 250 --output runs/fb_four_rooms
```

To collect a *fixed* random replay dataset rather than using enumerated oracle
transitions, add `--dataset random --transitions 5000`. The data collection
process ends before offline training begins.

The script refuses to append into a nonempty output directory. Choose a new
`--output` path per experiment. It records:

- `manifest.json`: experiment specification and completion status/provenance;
- `dataset.npz`: exactly the training observations, actions and next observations;
- `training_metrics.jsonl`: typed metric events indexed by gradient updates;
- `evaluations.jsonl`: matched-seed initial and final goal evaluation;
- `checkpoint.pt`: agent/optimizer/agent RNG state and sampler RNG state.

`dataset.npz` can be loaded with `numpy.load(..., allow_pickle=False)`. Do not
load `checkpoint.pt` from untrusted sources: PyTorch checkpoint formats require
careful deserialization practices.

## Clear methodological limitations

This is a *diagnostic* Four Rooms layout designed for AERIS, **not** the precise
maze environment or complete training recipe used in the original publication.
The `oracle` dataset provides all traversable state-action pairs, and is therefore
not a demonstration of how random exploration covers the environment.
The environment is reward-free; goal success is assessed externally. The policy
receives only a goal condition obtained from `B(goal)`; no evaluation trajectory
rewards are supplied for condition inference.

Expected performance is not guaranteed after only 250 updates. Even if the
results show improved success, this alone is **not** reproduction of published
maze scores, nor proof of zero-shot generalization beyond this test case.

## Deliberate separation

- `environments/four_rooms.py`: Four Rooms dynamics, space spec, goal tasks and exact tabular model; `environments/tabular.py`: `TabularDynamics` and `TabularMDP`.
- `experiments/collection.py`: generic environment → transitions.
- `agents/fb/batch.py`: FB-specific training batch sampling (already exists).
- `experiments/training.py`: generic BatchProvider → TrainableAgent updates.
- `experiments/evaluation.py`: generic agent actions → goal success metrics.
- `data/run_directory.py`: `RunDirectory`, typed reading and writing of the manifest, metrics, evaluations, trajectories, policy snapshots and reports.
- `data/artifacts.py`: NumPy formats for policy snapshots and the fingerprinted transition dataset.
- `examples/fb_four_rooms.py`: **composition root**; selects FB and FourRooms.

The runner and evaluator never import FB or FourRooms. That is the first
architectural interchangeability demonstration. Tests also exercise the runner
and evaluator with a non-FB dummy agent.

## Next scientific tests

1. Run several training/evaluation seeds and report confidence intervals.
2. Compare to random-policy, shortest-path and value-iteration baselines.
3. Train on controlled coverage subsets and measure policy degradation.
4. Check exact FB objective and target-network update against pinned upstream.
5. Reproduce the original benchmark protocol only after aligning environment,
   state encoding, dataset collection, and evaluation details.

### Fidelity correction discovered during integration

The earlier AERIS `DiscreteFBAgent` used row-aligned broadcasting
`b_b_detach[:, None]` in the regularizer. The original grid agent multiplies
its `[batch]` vector directly into a `[batch, batch]` matrix, which aligns it
to *columns* in PyTorch. This integration adjusts the expression to
`b_b_detach[None, :]` and updates the reference-formula unit test. It therefore
**replaces** two prior files: `agents/fb/agent.py` and
`tests/agents/test_discrete_fb.py`.

Other differences from upstream remain intentionally visible. In particular,
our goal evaluation measures greedy rollout success in this new environment;
the paper's grid evaluation computes value-based scores and may use generalized
policy improvement. Original FB also groups target updates with its training
cycles, while the AERIS agent currently Polyak-updates each gradient step.

On the provided 250-update diagnostic check, final success was 0.0 on 12
seeded evaluation episodes. That should trigger proper algorithmic validation
rather than be interpreted as successful learning. The 55 passing tests certify
integration semantics and formulas, **not policy quality**.
