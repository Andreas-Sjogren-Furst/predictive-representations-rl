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
  `dashboards/` (commit them to keep figure setups reproducible). Two layouts are built in:
  *Run overview*, *Zero-shot suite*, *Representations* and *Compare runs*; a saved layout with the same name replaces a built-in one.
- **Live refresh** (sidebar toggle): re-reads runs every few seconds, to follow a training run.
- **Runs tab:** every run under the runs directory with its status, agent and seed.

A panel whose data is missing says so and names the command that creates it, for example
`aeris analysis zero-shot runs/…`.

## Panels

| Panel | Scope | Shows |
|---|---|---|
| `run_comparison` | many runs | settings and key results of each selected run side by side |
| `loss_curves` | many runs | any logged training metric, with optional smoothing; overlays runs |
| `evaluation` | many runs | recorded evaluation metrics per checkpoint |
| `policy_map` | one run | the saved action in each cell, coloured by exact optimality |
| `trajectory` | one run | an animated replay of a recorded evaluation episode |
| `suite_scores` | many runs | zero-shot mean normalized score per task family |
| `suite_distribution` | many runs | every task's zero-shot score, grouped by family |
| `task_policy` | one run | one task's reward map with the action the agent inferred in every cell |
| `goal_scores_map` | one run | zero-shot score or goal success per goal cell |
| `suite_tasks` | one run | a table of every suite task's results |
| `report` | one run | the numbers of any saved analysis report |
| `run_overview` | one run | status, provenance and every recorded setting |
| `projection` | one run | PCA (2-D or 3-D) of each state's representation, coloured by room, distance to goal, row, column or state density |
| `similarity` | one run | similarity between every pair of states' representations, ordered by room |
| `b_spectrum` | one run | singular values of B and its effective rank |
| `occupancy_maps` | one run | learned F(s0, a, z)·B next to the exact occupancy density m/ρ, for a chosen goal, start cell and first action |
| `value_map` | one run | learned max_a F·z next to the exact optimal value V* |

## Representation panels

The last five panels read `analysis/fb_representations_v1.npz`, created once per run with

```bash
aeris analysis representations runs/<name>          # all goals; or --goal 9,9 --goal 1,1
```

It loads the checkpoint (torch) and saves, for every state, B(s), and for every goal z = B(goal):
F(s, a, z), the target policy and the exact occupancy density. The density is divided by ρ, the
frequency of each cell in the run's own training data, which is what FB's F·B estimates, so the
comparison is valid for random as well as oracle datasets. For Four Rooms with all 68 goals it takes
about 15 seconds and 8 MB. The built-in *Representations* layout shows all five panels.

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
Plotly figure, a `Table` or a `Message`. It never imports Streamlit and never reads files
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
