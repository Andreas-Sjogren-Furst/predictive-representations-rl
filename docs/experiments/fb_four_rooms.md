# First end-to-end discrete FB diagnostic

This integration demonstrates the **existing AERIS contracts** working with a
real discrete FB learner and a four-room grid-world environment.

## Run

```bash
uv sync --extra fb-torch
uv run --extra fb-torch pytest -q
uv run --extra fb-torch aeris train four-rooms --updates 250 --output runs/fb_four_rooms
```

To collect a *fixed* random replay dataset rather than using enumerated oracle
transitions, add `--dataset random --transitions 5000`. The data collection
process ends before offline training begins.

## Dataset scenarios

The oracle dataset can leave parts of the grid out, to test what FB learns from
data that does not cover everything. `--scenario NAME` picks one of the exact
datasets in `environments/four_rooms_datasets.py` (default `full`, every
state-action pair once):

```bash
aeris datasets list                                             # every scenario and its coverage
aeris train four-rooms --scenario rooms_no_bottom_right --updates 40000
aeris sweep four-rooms --name fb_no_bottom_right --seeds 0-4 --lsf --scenario rooms_no_bottom_right --updates 40000
```

| Scenario | Start states | State-action pairs | Transitions | What it leaves out |
|---|---|---|---|---|
| `full` | 68 | 272 | 272 | nothing |
| `room_top_left` | 16 | 64 | 64 | every room but the top-left |
| `rooms_top` | 33 | 132 | 132 | the bottom rooms |
| `rooms_no_bottom_right` | 52 | 208 | 208 | the bottom-right room, where the default goal (9,9) is |
| `no_left` | 68 | 204 | 204 | the action ← |
| `no_doorway_moves` | 64 | 248 | 248 | every move into or out of a doorway |
| `skew_top_left` | 68 | 272 | 848 | nothing; the top-left room's pairs appear 10 times |
| `rare_doorways` | 68 | 272 | 2,576 | nothing; every pair 10 times, the doorways' once |
| `no_goal` | 67 | 268 | 268 | the goal (9,9) as a start |
| `closed_doorway` | 67 | 268 | 268 | the doorway (8,5) between the bottom rooms as a start |
| `holes_25` | 67 | 204 | 204 | a random 25% of the pairs (seeded) |
| `holes_50` | 63 | 136 | 136 | a random 50% of the pairs (seeded) |

Every transition is the true one, and each covered pair appears a fixed number of
times, so a scenario changes only coverage, never adds noise. Moves out of the
covered states are kept: a cell can be a next state without ever being a start
(the doorways of `room_top_left`, the goal in `no_goal`). The manifest records
the scenario and its full spec under `dataset.scenario` and `dataset.spec`, so
`aeris runs summary --group-by dataset.scenario` and the dashboard's *Group runs
by* compare scenarios directly. New scenarios are one `OracleDataset(...)` line
in `PRESETS`: rooms, extra or dropped cells, dropped actions, moves to drop,
copies per area and a random share of holes.

On partial data, the zero-shot suite's `dataset_rewards` evidence only sees the
covered states, so some tasks have no rewarded sample at all (the dashboard's
`dataset_coverage` panel counts them per family). The FB occupancy diagnostic
(`aeris analysis fb-checkpoint`) needs every pair exactly once, so it runs only
for `full`.

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
- `workflows/four_rooms_training.py`: **composition root**; selects FB and FourRooms, records the full agent configuration in the manifest. `aeris train four-rooms` and `examples/fb_four_rooms.py` both call it.

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
