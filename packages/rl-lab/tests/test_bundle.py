import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from rl_lab.bundle.io import content_hash, write_yaml
from rl_lab.bundle.reader import load_manifest, load_view, validate_bundle, validate_dataset
from rl_lab.bundle.writer import write_dataset, write_manifest, write_view
from rl_lab.contracts import DatasetSpec, Manifest

N = 30


def dataset_spec(num=N):
    return DatasetSpec.model_validate({
        "id": "toy-probes", "hash": "",
        "source": {"kind": "held_out_dataset", "id": "toy-val"},
        "construction": {"method": "random_windows", "num_samples": num, "window_length": 4, "seed": 0},
        "labels": {"x": {"kind": "continuous", "unit": "m"}, "room": {"kind": "categorical", "classes": 2}},
        "grouping": {"column": "episode_id"},
    })


def samples(num=N):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "sample_id": [f"s{i:03d}" for i in range(num)],
        "episode_id": [f"e{i // 5}" for i in range(num)],
        "timestep": np.arange(num) % 5,
        "x": rng.uniform(-1, 1, num),
        "room": rng.integers(0, 2, num),
    })


def manifest_dict(dataset_hash, views=None):
    return {
        "schema_version": "1.0",
        "exporter": {"name": "test", "version": "0"},
        "experiment": {
            "id": "toy-exp", "group": "toy", "seed": 0,
            "algorithm": {"name": "toy-algo"},
            "environment": {"id": "toy-env", "tasks": ["a"],
                            "task_definition": {"source": "toy.py", "fingerprint": "sha256:abc"}},
            "training": {"mode": "offline", "objective": "reward_free",
                         "data": {"kind": "dataset", "id": "toy-data"}, "budget": {"gradient_steps": 10}},
        },
        "views": views if views is not None else [
            view("z", dataset_hash),
            view("z__untrained", dataset_hash, role="control", control_kind="untrained", control_of="z", step=0),
        ],
    }


def view(view_id, dataset_hash, role="trained", step=10, **control):
    return {"id": view_id, "role": role, "checkpoint_step": step, "checkpoint_unit": "gradient_steps",
            "dataset": {"id": "toy-probes", "hash": dataset_hash}, "path": f"representations/{view_id}.npz",
            "protocol": {"input": "final_state"}, **control}


@pytest.fixture
def catalog(tmp_path):
    spec = write_dataset(tmp_path / "datasets" / "toy-probes", dataset_spec(), samples())
    bundle = tmp_path / "experiments" / "toy-exp"
    ids = samples()["sample_id"].to_numpy()
    rng = np.random.default_rng(1)
    write_view(bundle, "representations/z.npz", ids, rng.normal(size=(N, 3)))
    write_view(bundle, "representations/z__untrained.npz", ids, rng.normal(size=(N, 3)))
    write_manifest(bundle, Manifest.model_validate(manifest_dict(spec.hash)))
    return tmp_path, bundle, spec


def test_valid_catalog_round_trips(catalog):
    root, bundle, spec = catalog

    assert validate_dataset(root / "datasets" / "toy-probes").ok
    report = validate_bundle(bundle, root / "datasets")
    assert report.ok and report.warnings == []
    sample_ids, embeddings = load_view(bundle, load_manifest(bundle), "z")
    assert list(sample_ids[:2]) == ["s000", "s001"] and embeddings.shape == (N, 3)


def test_content_hash_depends_on_values_not_on_order_of_names():
    a = {"x": np.arange(3.0), "ids": np.array(["a", "b", "c"])}
    assert content_hash(a) == content_hash(dict(reversed(list(a.items()))))
    assert content_hash(a) != content_hash({**a, "x": np.array([0.0, 1.0, 2.5])})
    assert content_hash(a) != content_hash({**a, "ids": np.array(["a", "b", "d"])})


def test_dataset_cannot_be_overwritten_with_different_content(catalog):
    root, _, _ = catalog
    changed = samples().assign(x=0.0)

    with pytest.raises(ValueError, match="different content"):
        write_dataset(root / "datasets" / "toy-probes", dataset_spec(), changed)


def test_changed_dataset_files_are_detected(catalog):
    root, bundle, _ = catalog
    samples().assign(x=1.0).to_parquet(root / "datasets" / "toy-probes" / "samples.parquet", index=False)

    assert "content hash mismatch" in " ".join(validate_dataset(root / "datasets" / "toy-probes").errors)


def test_bundle_must_match_its_dataset_hash(catalog):
    root, bundle, spec = catalog
    write_manifest(bundle, Manifest.model_validate(manifest_dict("sha256:other")))

    assert "has changed since export" in " ".join(validate_bundle(bundle, root / "datasets").errors)


def test_view_sample_ids_must_belong_to_the_dataset(catalog):
    root, bundle, _ = catalog
    write_view(bundle, "representations/z.npz", np.array([f"x{i}" for i in range(N)]), np.zeros((N, 3)))

    assert "not in dataset" in " ".join(validate_bundle(bundle, root / "datasets").errors)


def test_non_finite_and_pickled_views_are_rejected(catalog):
    root, bundle, _ = catalog
    ids = samples()["sample_id"].to_numpy().astype(str)
    np.savez(bundle / "representations" / "z.npz", sample_ids=ids, embeddings=np.full((N, 3), np.nan))
    np.savez(bundle / "representations" / "z__untrained.npz", sample_ids=ids.astype(object), embeddings=np.zeros((N, 3)))

    errors = " ".join(validate_bundle(bundle, root / "datasets").errors)
    assert "non-finite" in errors
    assert "cannot read representations/z__untrained.npz" in errors  # object arrays need pickle, which is refused


def test_paths_may_not_escape_the_bundle(catalog):
    root, bundle, spec = catalog
    data = manifest_dict(spec.hash, views=[view("z", spec.hash) | {"path": "../../secret.npz"}])
    write_yaml(bundle / "manifest.yaml", data)

    assert "escapes the bundle" in " ".join(validate_bundle(bundle, root / "datasets").errors)


def test_missing_controls_and_task_definition_are_warnings(catalog):
    root, bundle, spec = catalog
    data = manifest_dict(spec.hash, views=[view("z", spec.hash)])
    del data["experiment"]["environment"]["task_definition"]
    write_manifest(bundle, Manifest.model_validate(data))

    report = validate_bundle(bundle, root / "datasets")
    assert report.ok
    assert any("task-definition" in w for w in report.warnings)
    assert any("no untrained control" in w for w in report.warnings)


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(schema_version="2.0"), "unsupported schema_version"),
    (lambda d: d["views"].append(dict(d["views"][0])), "duplicate view ids"),
    (lambda d: d["views"][1].update(control_of="nope"), "unknown view"),
    (lambda d: d["views"][1].pop("control_kind"), "control views need"),
    (lambda d: d["experiment"]["training"].update(objective="task_reward"), "requires trained_task"),
    (lambda d: d.update(surprise=1), "Extra inputs"),
])
def test_invalid_manifests_are_rejected(change, message):
    data = manifest_dict("sha256:x")
    change(data)

    with pytest.raises(ValidationError, match=message):
        Manifest.model_validate(data)
