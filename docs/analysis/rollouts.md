# Exact rollouts and policy fates

On Four Rooms the dynamics are deterministic and a zero-shot policy acts greedily, so a policy
gives every state exactly one successor. Its whole future from any start is therefore known
exactly: a path that either enters the goal or falls into a **cycle** it never leaves. No episode
needs to be sampled or recorded; `analysis/rollouts.py` computes the path from one action per
state (`TabularDynamics` in, plain records out).

```python
from aeris.analysis.rollouts import policy_fates, rollout

path = rollout(dynamics, actions, start, goal=goal)  # stops at the goal or the first repeat
path.states, path.actions, path.goal_step, path.cycle
fates = policy_fates(dynamics, actions, goal=goal)  # every start at once
fates.goal_outcomes(max_steps=60)  # at goal / optimal / detour / late / trapped
```

- **`rollout`** follows the policy until it enters the goal (goal tasks) or reaches a state it has
  already visited. In the second case the last state repeats `states[cycle_start]` and `cycle`
  lists the states it loops through forever. A one-state cycle means every step bumps into a wall.
- **`policy_fates`** rolls out from every start: steps to the goal (-1 if never), the cycle each
  start ends in (cycles are deduplicated), the steps before it gets there, and `visits[s]`, the
  number of *other* starts whose path enters `s` (where paths converge).
- **`goal_outcomes`** compares each start with the shortest path: *optimal* (fewest possible
  steps), *detour* (more steps, still within `max_steps`), *late* (a detour beyond `max_steps`),
  *trapped* (a cycle that never reaches the goal). Goal success in the zero-shot suite is
  exactly the share of non-goal starts that are optimal or detour.

`analysis/task_scores.py` adds two exact references per start:

- `optimal_solution(dynamics, rewards, gamma)`: V\*, Q\* and the set of optimal actions in every
  state, for the suite's continuing objective. Used to mark suboptimal actions along a path and to
  draw an optimal path.
- `start_scores(dynamics, actions, rewards, gamma)`: the normalized score from each start alone,
  `(V^π(s) − V^rand(s)) / (V*(s) − V^rand(s))`. 1 is optimal from that start, 0 is no better than
  acting at random, below 0 is worse. The suite's score normalizes the *means* over starts, so it
  is not the mean of these.

## In the dashboard

The `rollout` and `fate_map` panels (built-in layout *Behaviour*) apply these functions to the
actions that `aeris analysis zero-shot` saved for every suite task and evidence protocol. They
work on any run with a suite report, need no checkpoint or torch, and render in well under a
second.

- **`rollout`** replays one task from one start cell, step by step. It overlays an optimal path
  (dashed) and, optionally, the other evidence protocol's path (dotted), and marks every
  suboptimal action with ×. The background is the task's reward (on hover). The subtitle gives
  the outcome, the number of suboptimal actions and the start score.
- **`fate_map`** colours every start cell for one task:
  - *outcome*: for goal tasks, optimal / detour / late / trapped. For other tasks, the loop each
    start ends in: the three loops that catch the most starts get their own colour, and the rest
    are shown as "other". Each colour-bar entry gives that loop's mean reward per step, and the
    subtitle compares the best of them with the best loop of an optimal policy.
  - *start score*: red below random, grey at random, blue at optimal.
  - *visits*: how many other starts pass through each cell, showing where paths converge.

  Loops are outlined in ink, and hovering a cell describes its fate.

Because the dynamics are deterministic, these rollouts are exactly what the evaluation would
see. `trajectories.jsonl` (sampled episodes recorded during training) remains the record for
environments where this does not hold.
