# AERIS — Exact tabular policy diagnostics (v1)

This step is an **additive** analysis package. It reads the existing Four Rooms `manifest.json` and `policy_initial.npz`/`policy_final.npz`, and does not alter the training process, FB networks, environment adapter, or 61 passing tests.

## Scientific definition

The diagnostic is an *exact deterministic discounted MDP* whose states are traversable Four Rooms cells and actions are the environment's four movement actions. The goal reward is **1 on entering the selected goal cell, 0 otherwise**, with a terminal goal state. Training experience remains reward-free. The default discount is `gamma=0.99`, deliberately an **analysis setting**, not assumed to be identical to the FB learner's training discount. Specify the agent's relevant discount with `--gamma` when appropriate; this does not rerun training.

For nonterminal states, value iteration solves

`Q*(s,a) = r(s,a) + gamma * V*(next(s,a))`, `V*(s) = max_a Q*(s,a)`.

The known goal and deterministic transition graph let us check the solution against exact shortest-path distances (`V*(s) = gamma^(d(s,goal)-1)` for non-goal states when `0 < gamma < 1`).

Actions within `1e-8` of the maximum optimal Q-value are all counted as optimal, including ties. Goal states are excluded from statistics, since the task ends there.

**Outputs, per saved snapshot:**

- `optimal_action_fraction`: fraction of all non-goal traversable states at which the selected action is optimal.
- `mean_action_regret`, `maximum_action_regret`: gap `max_a Q*(s,a)-Q*(s,selected_action)` under the specified discounted reward (not an empirical return).
- `rollout_success_fraction`: exact deterministic success across **every non-goal start cell**, with the evaluation's recorded finite step horizon. Cycles fail rather than being silently treated as eventual successes.
- `mean_excess_steps_on_success`: successful deterministic rollout length minus exact shortest-path length, averaged over successful starts (null if none succeeds).
- `success_gap_to_optimal`: difference between the optimal policy's and saved policy's all-state success fractions, evaluated at the **same horizon**.

A full-coverage, uniformly weighted set of start states is **not** the same distribution as the recorded finite-episode `success_rate`. The two statistics must not be presented as interchangeable.

`maximum_shortest_path_steps` is emitted because a short evaluation horizon can make even an optimal policy fail from some start cells. These policy diagnostics evaluate action and behavior quality; they do not directly establish why F and B representations are inaccurate.

## Usage

From the AERIS repository root, after installing the existing `fb-torch` and `viz` extras:

```bash
uv run --extra fb-torch --extra viz python examples/analyze_four_rooms.py runs/fb_four_rooms_visual
```

No Matplotlib required for numeric analysis:

```bash
uv run --extra fb-torch python examples/analyze_four_rooms.py runs/fb_four_rooms_visual --no-plots
```

`--gamma 0.99` is the default. Supply another value strictly between 0 and 1 if desired (0 is also mathematically supported).

Generated files:

```text
runs/fb_four_rooms_visual/
  analysis/tabular_policy_v1.json
  figures/optimal_policy.png
  figures/policy_gap_initial.png
  figures/policy_gap_final.png
```

The images distinguish optimal (green) from suboptimal (red) selected actions and show all tied optimal actions in the oracle plot. Everything comes from the saved policy snapshots; **no checkpoint is deserialized and no gradient update is performed**. The training manifest and saved results remain unmodified.

## Scope and limitations

- Four Rooms layout and tabular transitions are deterministic and known. The general tabular solver is environment-independent, but the adapter and plots are Four Rooms-specific.
- This is a *policy-quality* diagnostic. FB forward/backward representations are not directly tested here.
- For cross-task or cross-environment comparisons, the reward, gamma, horizon and reset-state weighting must be specified consistently.
- The study's original published FB maze benchmark remains a **separate reproducibility milestone**; these are AERIS diagnostic environments and tasks.
- No Streamlit dependency has been introduced. The core tabular analysis is graphics-free and PyTorch-free.

## Verification

With all previous AERIS integrations applied:

```bash
uv run --extra fb-torch --extra viz pytest -q
# Expected: 70 tests (61 prior + 9 new)
```
