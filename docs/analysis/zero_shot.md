# Zero-shot task suites

Zero-shot RL claims that one pretrained representation solves *new* tasks without further
learning, given only a little evidence about each task's reward. AERIS tests this claim on **task
suites**: named, versioned sets of tasks on which every agent is evaluated identically, so that FB,
one-step FB and TD-JEPA can be compared on equal terms.

```bash
aeris analysis zero-shot runs/<name>                   # suite four_rooms_v1, all pretraining data
aeris analysis zero-shot runs/<name> --samples 20      # only 20 relabelled observations as evidence
```

The report is saved as `analysis/zero_shot_suite_v1.json` and shown by the dashboard's
*Zero-shot suite* layout. On Four Rooms the whole suite takes about 20 seconds.

## The four_rooms_v1 suite

A task is a `TaskSpec` whose `settings["type"]` names its reward family; the environment turns it
into a reward r(s) for every cell (`FourRooms.state_rewards`, built by
`environments/four_rooms_tasks.py`).

| Family | Tasks | Reward |
|---|---|---|
| `goal` | 68, one per cell | 1 in the goal cell |
| `room` | 4 | 1 in every cell of one room (doorways excluded) |
| `distance` | 8 (centre and outer corner of each room) | −(shortest-path distance to the goal) / (largest distance): dense |
| `random_sparse` | 20 (seeds 0–19) | 1 in 3 random cells: several goals at once |
| `random_dense` | 20 (seeds 0–19) | an independent standard normal per cell |

Goal tasks saved before task types existed (settings with only `goal_row`/`goal_col`) still read as
goal tasks.

## Protocol

For every task and every evidence protocol:

1. **Infer** a task condition once, through the `TaskConditioner` contract (FB: `z = mean_i r(s_i) B(s_i)`).
2. **Act**: query the agent's greedy action in every cell through the `Agent` contract, so any
   zero-shot agent can be evaluated without suite changes.
3. **Score exactly** with the known dynamics (`analysis/task_scores.py`), no sampling:
   - objective: the discounted sum of state rewards, current state included, under a continuing
     policy, `V(s) = r(s) + γ V(next(s))`, with the agent's γ; this is what FB's `F·z` estimates;
   - V^π of the agent's policy, V* of the optimal policy and V^rand of the uniformly random policy,
     each averaged uniformly over start cells;
   - **normalized score = (V^π − V^rand) / (V* − V^rand)**: 0 is no better than random, 1 is optimal,
     and it stays meaningful for negative rewards;
   - goal tasks also report **goal success**: the fraction of non-goal start cells from which the
     policy first reaches the goal within the run's evaluation horizon.

| Evidence protocol | Evidence | Role |
|---|---|---|
| `dataset_rewards` | the run's reward-free pretraining observations, relabelled with r(s) | the **zero-shot** setting |
| `all_states` | every cell once with its reward | **privileged**: the whole reward function, independent of what the data covered |

On an oracle dataset (every cell exactly 4 times) both protocols infer the same z and score
identically; on other datasets the gap shows how data coverage affects inference. With `--samples N`
many tasks may have **no rewarded evidence** (`rewarded_samples = 0`): their inferred z is zero, and
the dashboard marks them.

Evidence never comes from evaluation; the agent is never updated.

## First results (Four Rooms, discrete FB, 40k updates, one seed)

| Family | oracle data | random data |
|---|---|---|
| all (120) | 0.850 | 0.853 |
| goal | 0.882 | 0.886 |
| room | 0.958 | 0.912 |
| distance | 0.876 | 0.916 |
| random_sparse | 0.889 | 0.894 |
| random_dense | 0.669 | 0.664 |

Mean normalized score with `dataset_rewards` evidence. Dense random rewards, which do not resemble
goals, are clearly the hardest family. These are single-seed numbers.

## Code

| Layer | Piece |
|---|---|
| `domain/task.py` | `TaskSuite`, `task_family` |
| `environments/four_rooms.py`, `four_rooms_tasks.py` | `state_rewards`, task builders, `four_rooms_v1` |
| `contracts/environment.py` | `TabularModel.state_rewards` |
| `analysis/task_scores.py` | exact `policy_values`, `optimal_values`, `task_baseline`, `score_policy` |
| `domain/reports.py` | `ZeroShotSuiteReport`, `SuiteTaskResult`, `SuiteOutcome` (incl. the greedy action per cell) |
| `workflows/four_rooms.py` | `evaluate_task_suite` |
| `dashboard/panels/tasks.py` | score by family, score distribution, task policy, goal map, task table |
