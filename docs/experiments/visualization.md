# AERIS — Four Rooms observability (diagnostic MVP)

This change adds **optional** evaluation trajectory capture and static figures. It
is **not** a new training algorithm or a paper replication.

## Data flow

- `evaluate_goal_policy(..., on_trajectory=..., max_trajectories=N)` emits
  bounded `EvaluationTrajectory` records, with actual observations, actions,
  next observations, rewards if present, termination/truncation, success,
  checkpoint identity, task identity, episode index and seed. The hook is
  optional and does not affect policy selection or aggregate success rates.
- `RunDirectory.write_trajectory` writes `trajectories.jsonl` separately
  from `evaluations.jsonl`. The JSON format is suitable for **small vector
  observations**. Do not use JSONL for large pixel frames; use a separate
  binary format and link artifacts in a future schema.
- A Four Rooms policy snapshot records each traversable state and its
  deterministic chosen action to `policy_initial.npz` and `policy_final.npz`.
  These are **behavioral** samples from the public agent interface, not
  ground-truth optimal policies or internal FB values.
- The visualization module reads previously recorded artifacts and creates
  PNG images using matplotlib's headless `Agg` backend. It does **not** require
  a model checkpoint or GPU. It understands Four Rooms coordinates; the
  generic evaluator/recorder do not.

## How to run

Install plotting support in your project (see `PYPROJECT_SNIPPET.toml`).

```bash
uv sync --extra fb-torch --extra viz
uv run --extra fb-torch --extra viz python examples/fb_four_rooms.py \
    --dataset oracle --updates 250 --episodes 20 \
    --trajectories 3 --plots --output runs/fb_four_rooms_visual
```

To regenerate PNGs from a completed run without retraining:

```bash
uv run --extra viz aeris runs plot runs/fb_four_rooms_visual
```

The report saves `figures/training.png`, `figures/success_rate.png`, `figures/policy_initial.png`,
`figures/policy_final.png` and (when traces are saved) first-episode
`figures/trajectory_initial.png` and `figures/trajectory_final.png`.
The full recorded sample of episodes remains available in `trajectories.jsonl`.

## Scientific limitations

- Deterministic evaluation conditions are inferred only from the chosen goal
  observation and a checkpoint; evaluation rollouts are **never** used for
  training or task inference.
- Identical evaluation seeds are used before and after training.
- A policy arrow plot shows **selected actions**; it does not establish
  optimality. An optimal-policy comparator is a separate future analysis.
- Only a subset of evaluation episodes is recorded to control artifact size.
  Success-rate metrics still use the **full** evaluation episode count.
- `npz` files are loaded with `allow_pickle=False`.
- Existing experiments recorded before this patch may have no saved snapshots
  or trajectories. The renderer can still produce whichever artifacts exist.
- The example still uses a synthetic Four Rooms layout; reproducing the
  original paper's maze benchmark remains a separate validation task.
