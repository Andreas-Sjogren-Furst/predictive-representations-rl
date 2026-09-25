from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FB_ROOT = ROOT / "third_party" / "onestep-fb"

sys.path.insert(0, str(FB_ROOT))

from utils.datasets import Dataset, GCDataset  # noqa: E402


def main() -> None:
    raw = {
        "observations": np.array(
            [
                [0.0],
                [1.0],
                [100.0],
                [101.0],
                [102.0],
            ],
            dtype=np.float32,
        ),
        "next_observations": np.array(
            [
                [1.0],
                [2.0],
                [101.0],
                [102.0],
                [103.0],
            ],
            dtype=np.float32,
        ),
        "actions": np.array(
            [
                [0.1],
                [0.2],
                [0.3],
                [0.4],
                [0.5],
            ],
            dtype=np.float32,
        ),
        "terminals": np.array(
            [0.0, 1.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        ),
    }

    dataset = Dataset.create(**raw)

    config = {
        "discount": 0.99,
        "frame_stack": None,
        "relabeling": False,
        "value_p_curgoal": 0.2,
        "value_p_trajgoal": 0.5,
        "value_p_randomgoal": 0.3,
        "value_geom_sample": True,
        "actor_p_curgoal": 0.0,
        "actor_p_trajgoal": 1.0,
        "actor_p_randomgoal": 0.0,
        "actor_geom_sample": False,
        "gc_negative": False,
        "p_aug": 0.0,
    }

    fb_dataset = GCDataset(dataset, config)

    batch = fb_dataset.sample(
        batch_size=4,
        augmentation=False,
    )

    print("FB dataset size:", fb_dataset.size)
    print("Batch keys:")

    for key, value in batch.items():
        shape = getattr(value, "shape", None)
        print(f"  {key:28s} {shape}")

    required = {
        "observations",
        "actions",
        "next_observations",
        "goals",
    }

    missing = required - set(batch)

    if missing:
        raise RuntimeError(
            f"FB batch is missing required fields: {sorted(missing)}"
        )

    print("\nFB dataset smoke test passed.")


if __name__ == "__main__":
    main()