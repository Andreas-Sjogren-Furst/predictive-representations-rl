# Run dashboard

An interactive, Streamlit-based dashboard for inspecting and comparing saved runs. It is read-only:
it displays what the run directories contain and never trains or analyzes.

## Start it

```bash
uv sync --extra dashboard            # or: uv tool install --editable ".[fb-torch,viz,dashboard]" --force
aeris dashboard                      # options: --root runs --dashboards dashboards --port 8501
```

Open `http://localhost:8501`. On a remote machine through VS Code, the port is forwarded
automatically (see the *Ports* tab). Run the dashboard on a login node only for viewing; it does no
heavy computation.

## Using it

- **Sidebar:** choose the runs directory, select one or more runs, and pick a dashboard layout.
- **Panels:** each card has a ⚙ menu with its options (metrics, smoothing, snapshot, checkpoint,
  evidence protocol, …). Single-run panels also let you choose which selected run they show, so one
  page can put two runs' policy maps side by side.
- **Edit layout** (sidebar toggle): add panels, change their width, move them up or down, remove
  them, set the number of columns, and **Save** the layout under a name. Layouts are saved as JSON in
  `dashboards/` (commit them to keep figure setups reproducible). Five layouts are built in:
  *Run overview*, *Zero-shot suite*, *Behaviour*, *Representations* and *Compare runs*; a saved
  layout with the same name replaces a built-in one.
- **Shared pickers:** a layout can share options between its panels. *Behaviour* shares the task
  and the evidence protocol: one picker above the grid sets them for every panel, so a whole page
  follows one task. In a saved layout's JSON, `"shared": {"task": null}` adds such a picker
  (`null` starts at the first panel's default; a saved value starts there).
- **Group runs by** (multi-run panels): aggregate seeds of one experiment, or compare groups by any
  setting that differs between the selected runs; see [seeds and sweeps](experiments/sweeps.md).
- **Live refresh** (sidebar toggle): re-reads runs every few seconds, to follow a training run.
- **Runs tab:** every run under the runs directory with its status, agent and seed.

The *Run overview* layout answers "did this run work, and where does it fail?" from the top down:
the summary tiles, the training goal's policy map next to an exact rollout, training curves next to
the zero-shot score per task family, then the analysis report and the run's settings. The policy map
uses the goal's own z = B(goal), as during training; the rollout and the suite use z inferred from
reward samples (`dataset_rewards` by default), so the two can differ.

A panel whose data is missing says so and names the command that creates it, for example
`aeris analysis zero-shot runs/…`.

## Panels

| Panel | Scope | Shows |
|---|---|---|
| `run_summary` | one run | stat tiles: status and setup, the training goal's exact success and optimal actions (with the change since the initial policy), occupancy correlation, zero-shot score, zero-shot goal success and the weakest task family |
| `run_comparison` | many runs | settings and key results of each selected run side by side |
| `group_summary` | many runs | mean ± std across the seeds of each group, for every headline number |
| `loss_curves` | many runs | any logged training metric, one chart per metric (or overlaid), with optional smoothing; overlays runs, or groups as mean ± std bands |
| `evaluation` | many runs | recorded evaluation metrics per checkpoint |
| `policy_map` | one run | the training goal's saved policy (z = B(goal), initial or final): the action in every cell, suboptimal ones white on ink, over the outcome from each start (optimal, detour, late, trapped) |
| `trajectory` | one run | an animated replay of a recorded evaluation episode |
| `suite_scores` | many runs | zero-shot mean normalized score per task family, then over all tasks; groups get ± std error bars |
| `suite_distribution` | many runs | every task's zero-shot score, grouped by family; or the difference between the two evidence protocols per task |
| `task_policy` | one run | one task's reward map with the action the agent inferred in every cell: suboptimal actions white on ink, the loops it ends in circled; the subtitle gives the share of starts that reach an optimal loop (one collecting as much reward per step as an optimal policy's best loop) |
| `goal_scores_map` | one run | zero-shot score or goal success per goal cell, printed in each cell |
| `rollout` | one run | the exact path of a zero-shot policy for any suite task from any start (by default the lowest-scoring one), animated, next to an optimal path; suboptimal actions marked |
| `fate_map` | one run | every start cell coloured by where the zero-shot policy takes it (optimal, detour, late, trapped; or which loop, ✓ marking optimal ones), by start score, by how many paths pass through, or by how much the start score changes with the other evidence protocol; optionally with the action in every cell |
| `suite_tasks` | one run | a table of every suite task's results, lowest score first, with the gap between evidence protocols |
| `report` | one run | the numbers of any saved analysis report, rounded |
| `run_overview` | one run | every recorded setting by section (run, experiment, dataset, agent, training, …); provenance on request; ⚠ marks uncommitted code, errors and unfinished runs |
| `projection` | one run | PCA (2-D or 3-D) of each state's representation, coloured by room, distance to goal, row, column or state density |
| `similarity` | one run | similarity between every pair of states' representations, ordered and labelled by room |
| `b_spectrum` | one run | singular values of centred B and its effective rank, out of the at most S − 1 directions S states can span |
| `occupancy_maps` | one run | learned F(s0, a, z)·B, the exact occupancy density m/ρ and their difference, for a chosen goal, start cell and first action; the title gives the learned share of the exact occupancy mass |
| `value_map` | one run | learned F·z next to the exact value of the same objective (the goal's occupancy over ρ), under FB's own policy or an optimal one, on one colour scale |
| `action_values` | one run | for z = B(goal): where the best learned action is not optimal, and the learned, same-policy and optimal value of every action in one cell (by default the costliest mistake) |

## Behaviour panels

`rollout`, `fate_map` and `task_policy` read the actions saved in the zero-shot suite report and
roll them out exactly on the known dynamics, for any task and any start; no checkpoint is loaded.
See [exact rollouts and policy fates](analysis/rollouts.md). Their task choice also offers
**lowest-scoring task**, resolved per run and evidence protocol, so a saved layout can always show
the current worst case. The rollout's start choice likewise offers **lowest-scoring start** (the
default): among the starts within 0.05 of the lowest start score, the one with the longest path,
so the replay shows how the policy goes wrong rather than a single bump into a wall.

The *Behaviour* layout is a deep dive into one task, chosen once at the top: the replay from the
lowest-scoring start, where every start ends up, the reward with the agent's actions, the score
from each start, where paths converge, and where the two evidence protocols lead to different
results. That last view compares start scores, not actions: two actions that are equally good
(common on distance and room tasks) are not a difference.

Reward maps colour by what the reward means: light to darker blue from the lowest to the highest
reward when rewards have one sign (goal, room, distance, random sparse), and red–grey–blue centred
on 0 when they take both signs (random dense). Scores use red (worse than random), grey (random)
and blue (optimal) everywhere.

On Four Rooms the two evidence protocols give the same policy for every goal task: the inferred z
differs only in scale. On the oracle dataset they agree on every task, because it covers every
state equally. Use the *difference* view of `suite_distribution` to see where they disagree.

## Representation panels

The last six panels read `analysis/fb_representations_v1.npz`, created once per run with

```bash
aeris analysis representations runs/<name>          # all goals; or --goal 9,9 --goal 1,1
```

It loads the checkpoint (torch) and saves, for every state, B(s), and for every goal z = B(goal):
F(s, a, z), the target policy and the exact occupancy density. The density is divided by ρ, the
frequency of each cell in the run's own training data, which is what FB's F·B estimates, so the
comparison is valid for random as well as oracle datasets. For Four Rooms with all 68 goals it takes
about 15 seconds and 8 MB. The built-in *Representations* layout shows them for one goal, chosen
once at the top (a shared picker).

All comparisons are in FB's own units: F(s, a, z)·B(t) estimates m(s, a, t)/ρ(t), and with
z = B(goal), F·z estimates m(s, a, goal)/ρ(goal), the discounted number of visits to the goal
(continuing, nothing terminal) divided by its frequency in the data. `value_map` and
`action_values` compare it with exactly that quantity, never with a terminal goal-reaching value,
which measures something else.

On the 40k-update runs these views show F·B is **too local**. From (1,1) with z = B(9,9), the
learned occupancy correlates 0.99 with the exact one but holds only about 36% of its mass, which is
missing mostly in the rooms reached later. The learned value of the goal under FB's own policy is
about 10% of the exact one, while still correlating 0.95. Far from the goal the learned values are
nearly flat, so the action ranking there rests on small differences.

`projection` and `similarity` can show three representations:

- **F(s, π, z)**, the forward vector averaged over the target policy (default);
- **F·B**, each state's learned occupancy profile over all states;
- **B(s)**, the backward vector. The orthonormality regularizer drives E[B Bᵀ] towards I, so B is
  close to orthogonal across states and its projections and similarities show little structure by
  design (on the 40k-update Four Rooms runs: mean off-diagonal cosine 0.002, 2–3% variance per
  principal component). The state structure lives in F and F·B (there, a state's nearest neighbour
  lies in the same room 94% of the time). Use B to check the regularizer, F and F·B to see what the
  representation learned.

## Adding a panel

A panel is a small class in `src/aeris/dashboard/panels/`: it declares its options as data and turns
typed run data (`RunData`: manifest, metrics, evaluations, trajectories, snapshots, reports) into a
Plotly figure, a `Table`, a row of `Stats` (headline tiles) or a `Message`. It never imports Streamlit and never reads files
(`tests/test_architecture.py` enforces both), so panels would survive a move to another front end.

```python
@dataclass(frozen=True)
class EntropyCurve(Panel):
    name = "entropy"
    title = "Target-policy entropy"
    scope = "runs"

    def render(self, runs, options):
        fig = go.Figure()
        for run in runs:
            points = [
                (r.step, m.value)
                for r in run.training
                for m in r.metrics
                if m.name == "target_policy_entropy"
            ]
            if points:
                fig.add_trace(
                    go.Scatter(x=[p[0] for p in points], y=[p[1] for p in points], name=run.label)
                )
        return STYLE.apply(fig, self.title)
```

Register it in `panels/__init__.py`; it then appears in *Add panel*. Panels that need expensive
computation (anything with torch) should read a saved analysis artifact rather than compute in the
dashboard.
