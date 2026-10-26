# AERIS

A modular research framework for reproducible experiments and analysis of learning agents in
reinforcement learning, focused on zero-shot RL with predictive representations. It currently
implements discrete-action Forward-Backward (FB) representations (Touati & Ollivier, 2021) on a
Four Rooms grid world. One-step FB and TD-JEPA are planned next.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
```

The core package needs only NumPy. Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `fb-torch` | PyTorch | training and loading FB agents |
| `viz` | Matplotlib | rendering figures |
| `dashboard` | Streamlit, Plotly | the interactive run dashboard (`aeris dashboard`) |

## Quick start

Train FB on Four Rooms and save a self-describing run directory:

```bash
uv run aeris train four-rooms --updates 250 --output runs/fb_four_rooms
```

Inspect the completed run without retraining:

```bash
uv run aeris runs list                                  # every run under runs/
uv run aeris runs show fb_four_rooms                    # manifest and latest metrics
uv run aeris runs plot fb_four_rooms                    # figures/*.png from recorded artifacts
uv run aeris analysis four-rooms fb_four_rooms          # saved policies vs the exact optimal policy
uv run aeris analysis fb-checkpoint fb_four_rooms       # FB checkpoint vs exact occupancies
uv run aeris analysis zero-shot fb_four_rooms           # infer every goal task from reward evidence
uv run aeris analysis representations fb_four_rooms     # save B, F and exact densities for the dashboard
uv run aeris dashboard                                  # browse and compare runs interactively
```

A run can be named by its directory, its folder name under `runs/` (`--root` to change), or its
unique `run_id`. Each command writes derived artifacts (`analysis/*.json`, `figures/*.png`) next to the original
run and never changes it. The manifest records the full agent configuration, so a run is
described by `manifest.json` alone. `examples/fb_four_rooms.py` shows the same workflow from Python,
and `examples/discrete_fb_smoke.py` is a minimal training example without an environment.

## Package layout

```text
src/aeris/
  domain/          Framework-independent records: experiments, transitions, results, validation
  contracts/       Protocols at component boundaries: Agent, Environment, BatchProvider, RunRecorder
  agents/fb/       Discrete-action Forward-Backward agent, networks and batch sampler
  environments/    Four Rooms grid world and exact tabular models (TabularDynamics, TabularMDP)
  experiments/     Generic collection, training and goal-evaluation loops
  data/            Run directories (typed read/write of every artifact), file formats, run catalog
  analysis/        Exact tabular baselines and run/checkpoint diagnostics (no file I/O, no plotting)
  workflows/       Open a run, check eligibility, run analyses, save reports, render figures
  cli/             The `aeris` command (Typer): browse runs and run workflows
  dashboard/       Streamlit run dashboard: framework-free panels, saved layouts, a thin app shell
  visualization/   Optional Matplotlib figures from saved artifacts
```

Dependencies point inwards: `domain` imports nothing else from AERIS, implementations depend on
`contracts` and `domain`, and nothing outside `visualization` imports Matplotlib.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy
```

CI (`.github/workflows/ci.yml`) runs the same checks on Python 3.11 and 3.12 for every push to
`main` and every pull request, with a CPU-only PyTorch build. mypy checks `src/` and `examples/`;
the protocol contracts are structural, so the type checker, not `isinstance`, is what verifies
that a component really satisfies one.

## Documentation

- [Run dashboard](docs/dashboard.md): viewing, comparing and laying out runs; adding panels
- [Architecture overview](docs/architecture/overview.md): layers, dependency rules, where validation lives, extending AERIS
- [Domain models](docs/architecture/domain_models.md), [component contracts](docs/architecture/component_contracts.md), and the founding [experiment lifecycle](docs/architecture/experiment_lifecycle.md) specification
- [Discrete FB algorithm](docs/algorithms/discrete_fb.md)
- [Four Rooms experiment](docs/experiments/fb_four_rooms.md) and [visualization](docs/experiments/visualization.md)
- [Tabular policy analysis](docs/analysis/tabular_policy.md), [FB checkpoint analysis](docs/analysis/fb_checkpoint.md) and [zero-shot goal reaching](docs/analysis/zero_shot.md)
