import csv
import json

import numpy as np

from predictive_representations_rl.analysis.base import load_extraction
from predictive_representations_rl.analysis.linear_probe import LinearProbeAnalyzer
from predictive_representations_rl.analysis.pca import PCAAnalyzer

RNG = np.random.default_rng(0)


def toy_metadata(num=600):
    x, y = RNG.uniform(-0.3, 0.3, (2, num))
    return {
        "x": x,
        "y": y,
        "room": (x >= 0).astype(np.int64) + 2 * (y < 0).astype(np.int64),
        "episode": np.repeat(np.arange(num // 20), 20),
        "source_index": np.arange(num),
    }


def test_load_extraction_keeps_only_per_probe_arrays(tmp_path):
    meta = toy_metadata(40)
    np.savez(tmp_path / "metadata.npz", **meta)
    np.savez(tmp_path / "backward.npz", values=np.ones((40, 3)))
    np.savez(tmp_path / "latent.npz", values=np.ones((4, 3)), tasks=np.array(list("abcd")))

    representations, metadata = load_extraction(tmp_path)

    assert list(representations) == ["backward"]
    assert set(metadata) == set(meta)


def test_analyzers_write_their_outputs(tmp_path):
    meta = toy_metadata()
    representations = {"rep": np.column_stack([meta["x"], meta["y"], RNG.normal(size=600)])}

    pca_summary = PCAAnalyzer().run(representations, meta, tmp_path / "pca")
    probe_summary = LinearProbeAnalyzer().run(representations, meta, tmp_path / "linear_probe")

    assert {p.name for p in (tmp_path / "pca").iterdir()} == {"rep.npz", "map_rep.png", "explained_variance.png", "summary.json"}
    assert pca_summary["rep"]["dims"] == 3
    rows = list(csv.DictReader(open(tmp_path / "linear_probe" / "linear_probe.csv")))
    assert {row["factor"] for row in rows} == {"x", "y", "room"}
    assert probe_summary["rep"]["x"] > 0.99
    assert (tmp_path / "linear_probe" / "linear_probe.png").exists()
    assert json.loads((tmp_path / "linear_probe" / "summary.json").read_text()) == probe_summary
