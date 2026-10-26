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

## Quick start

Train FB on Four Rooms and save a self-describing run directory:

```bash
uv run python examples/fb_four_rooms.py --updates 250 --output runs/fb_four_rooms
```

Inspect the completed run without retraining:

```bash
uv run python examples/four_rooms_report.py render   runs/fb_four_rooms  # figures/*.png
uv run python examples/four_rooms_report.py analyze  runs/fb_four_rooms  # exact tabular policy analysis
uv run python examples/four_rooms_report.py diagnose runs/fb_four_rooms  # FB checkpoint vs exact occupancies
```

Each command writes derived artifacts (`analysis/*.json`, `figures/*.png`) next to the original
run and never changes it. `examples/discrete_fb_smoke.py` is a minimal training example without
an environment.

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
  visualization/   Optional Matplotlib figures from saved artifacts
```

Dependencies point inwards: `domain` imports nothing else from AERIS, implementations depend on
`contracts` and `domain`, and nothing outside `visualization` imports Matplotlib.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Documentation

- [Experiment lifecycle](docs/architecture/experiment_lifecycle.md), [domain models](docs/architecture/domain_models.md), [component contracts](docs/architecture/component_contracts.md)
- [Discrete FB algorithm](docs/algorithms/discrete_fb.md)
- [Four Rooms experiment](docs/experiments/fb_four_rooms.md) and [visualization](docs/experiments/visualization.md)
- [Tabular policy analysis](docs/analysis/tabular_policy.md) and [FB checkpoint analysis](docs/analysis/fb_checkpoint.md)
