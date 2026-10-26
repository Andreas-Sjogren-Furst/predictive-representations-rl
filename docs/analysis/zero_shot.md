# Zero-shot goal reaching (v1)

Zero-shot RL claims that one pretrained representation can solve a new task **without further
learning**, given only a little evidence about the task's reward. This evaluation tests that claim
on a completed Four Rooms FB run, for every goal cell.

## Protocol

For each goal task, and for each evidence protocol below, the agent infers its task condition once
through the `TaskConditioner` contract (`infer_condition`), then the frozen policy is evaluated on
seeded start states. All protocols use the **same evaluation seeds**, so start states are paired
and success rates can be compared goal by goal. The agent is never updated, and inference never sees
evaluation rollouts.

| Evidence protocol | Evidence given to inference | Role |
|---|---|---|
| `goal_observation` | one sample: the goal observation with reward 1 | **privileged** baseline: it names the goal state. For FB this is exactly `z = B(goal)`, the conditioning used by training's own evaluation |
| `dataset_rewards` | the run's reward-free pretraining observations, relabelled with the task reward `r(s) = 1` if `s` is the goal, else `0` | the **zero-shot** setting |

For FB, inference is `z = mean_i r(s_i) B(s_i)` over the evidence (see `DiscreteFBAgent.infer_condition`).

With `--samples N`, a fixed random subset of N pretraining observations is relabelled for every
task instead of all of them. Small N leaves many goals with **no rewarded sample**; their inferred
`z` is zero, which the report records as `rewarded_samples = 0` and the figure marks with ×.

## Run

```bash
uv run --extra fb-torch --extra viz aeris analysis zero-shot runs/fb_four_rooms
uv run --extra fb-torch aeris analysis zero-shot runs/fb_four_rooms --samples 20 --episodes 20 --no-plots
uv run --extra fb-torch aeris analysis zero-shot runs/fb_four_rooms --goal 9,9 --goal 1,1
```

Evaluating every goal with the default 10 episodes per goal and protocol takes about a minute on CPU.

Outputs (the original run is not modified):

- `analysis/zero_shot_goals_v1.json`: a typed `ZeroShotReport` with, per goal and protocol, the
  number of evidence and rewarded samples, success rate and mean episode length, plus the mean
  success per protocol and the definition of each protocol.
- `figures/zero_shot_success.png` (unless `--no-plots`): one success map per protocol.

## Checks and interpretation

- The privileged protocol reproduces training's own final evaluation exactly when given the same
  goal, episodes and seeds (a regression test enforces this), so the two evaluation paths agree.
- The inferred `z` scales with the fraction of rewarded samples (`z = (k/N) B(goal)` for `k`
  rewarded samples out of `N`). FB's forward network normalizes its condition input, but not in a
  scale-invariant way, so `dataset_rewards` can behave differently from `goal_observation` even with
  full coverage. That difference is part of what this evaluation measures; `z` is not renormalized.
- Success here is first-hitting the goal within the run's evaluation horizon from random start
  cells. Compare it with the exact optimum from `aeris analysis four-rooms` before reading a low
  rate as a representation failure.

Code: the generic inference-then-evaluation step is `experiments/zero_shot.py`
(`evaluate_zero_shot`, any `ZeroShotAgent`); evidence construction, eligibility and saving are in
`workflows/four_rooms.py` (`evaluate_zero_shot_goals`); the figure is `visualization/zero_shot.py`.
