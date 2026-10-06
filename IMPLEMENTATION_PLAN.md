# Lightweight RL Experiment Harness

## Goal

Build a **thin experimental harness** for comparing reinforcement-learning algorithms across environments without rewriting training or analysis code.

The core design principle is:

```text
Environment ⟂ Algorithm ⟂ Analysis
```

The framework should **not** replace Dreamer, FB, or other existing implementations. It should wrap them with small adapters and standardize only the boundaries needed for fair experiments and representation analysis.

A compatible algorithm or environment should be addable with **one small adapter**, without changing the rest of the system.

---

## 1. Minimal interfaces

### Environment adapter

```python
from abc import ABC, abstractmethod

class EnvAdapter(ABC):
    @abstractmethod
    def reset(self, seed: int | None = None):
        ...

    @abstractmethod
    def step(self, action):
        ...

    @property
    @abstractmethod
    def observation_spec(self):
        ...

    @property
    @abstractmethod
    def action_spec(self):
        ...

    def factors(self, observation) -> dict:
        """Optional ground-truth factors for analysis."""
        return {}
```

For Gymnasium environments, provide a generic `GymEnvAdapter`.

For controlled thesis environments such as Four Rooms, `factors()` should expose interpretable state variables such as:

```python
{
    "x": ...,
    "y": ...,
    "room": ...,
    "color": ...,
    "goal_distance": ...,
}
```

---

### Algorithm adapter

Do **not** force all algorithms into the same internal training loop.

Each adapter may call the original implementation internally.

```python
class AlgorithmAdapter(ABC):
    @abstractmethod
    def train(self, env, config):
        ...

    @abstractmethod
    def act(self, observation, eval: bool = False):
        ...

    @abstractmethod
    def save(self, path):
        ...

    @abstractmethod
    def load(self, path):
        ...

    def representations(self, batch) -> dict[str, "np.ndarray"]:
        """Return named representations for analysis."""
        return {}

    @property
    def capabilities(self) -> set[str]:
        return set()
```

Examples:

Dreamer:

```python
{
    "deterministic_state": h,
    "stochastic_state": z,
    "model_state": concat(h, z),
}
```

FB:

```python
{
    "backward": B,
    "forward": F,
}
```

All arrays returned to the analysis layer must be converted to **NumPy**.

---

## 2. Probe sets

Representation analysis must use a fixed set of states so algorithms are compared on exactly the same inputs.

```python
class ProbeSet:
    observations: np.ndarray
    metadata: dict[str, np.ndarray]
```

For small controlled environments, enumerate all states if possible.

Example metadata:

```python
{
    "x": ...,
    "y": ...,
    "room": ...,
    "reward": ...,
    "shortest_path_distance": ...,
}
```

The same probe set should be passed through Dreamer, FB, and future algorithms.

---

## 3. Analysis interface

Analysis modules should know nothing about the original algorithm implementation.

They receive:

```python
representations: np.ndarray   # shape [N, D]
metadata: dict
```

Initial analyses:

- PCA
- UMAP
- latent-dimension heatmaps
- linear probes
- pairwise latent-distance analysis
- CKA between checkpoints / tasks
- Procrustes or representational-similarity analysis

Example:

```python
class Analyzer(ABC):
    requires: set[str] = {"representation"}

    @abstractmethod
    def run(self, representations, metadata, output_dir):
        ...
```

Avoid comparing individual latent coordinates directly. Neural representations may rotate or permute while preserving the same information.

---

## 4. Suggested project structure

```text
src/
├── core/
│   ├── types.py
│   ├── registry.py
│   └── config.py
│
├── envs/
│   ├── base.py
│   ├── gym_adapter.py
│   └── four_rooms.py
│
├── algorithms/
│   ├── base.py
│   ├── dreamer_adapter.py
│   ├── fb_adapter.py
│   └── onestep_fb_adapter.py
│
├── probes/
│   ├── probe_set.py
│   └── extractor.py
│
├── analysis/
│   ├── pca.py
│   ├── umap.py
│   ├── linear_probe.py
│   ├── cka.py
│   ├── latent_distance.py
│   └── heatmap.py
│
├── runner.py
└── cli.py
```

Existing implementations remain under:

```text
third_party/
├── onestep-fb/
└── Offline_vs_Online_in_MBRL/
```

Adapters should wrap these repositories rather than heavily modifying them.

---

## 5. Standard experiment output

Every experiment should produce the same directory structure:

```text
runs/
└── <environment>/
    └── <algorithm>/
        └── seed_<N>/
            ├── config.json
            ├── metrics.jsonl
            ├── checkpoints/
            ├── probes/
            │   ├── <representation>.npz
            │   └── metadata.npz
            ├── videos/
            └── analysis/
```

Analysis code should read only these standardized artifacts, never algorithm-specific log directories.

---

## 6. CLI target

Desired usage:

```bash
python -m src.run     --env four_rooms     --algo dreamer     --seed 0
```

```bash
python -m src.run     --env four_rooms     --algo onestep_fb     --seed 0
```

Analysis:

```bash
python -m src.analyze     --run runs/four_rooms/dreamer/seed_0     --analysis pca,linear_probe,cka
```

---

## 7. First implementation milestone

Do not build every abstraction immediately.

Implement only enough to prove the design with:

1. One controlled environment: `FourRooms`.
2. `EnvAdapter`.
3. `AlgorithmAdapter`.
4. One Dreamer adapter.
5. One one-step-FB adapter.
6. `ProbeSet`.
7. PCA analysis.
8. Linear-probe analysis.
9. Standardized run directory.

Once both Dreamer and one-step FB can train and produce representations through the same pipeline, extend the framework.

The framework should stay **thin**. If an abstraction requires large changes to the original RL algorithms, it is probably too low-level.
