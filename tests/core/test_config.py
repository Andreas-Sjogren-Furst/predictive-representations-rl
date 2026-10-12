from pathlib import Path

import pytest

from predictive_representations_rl.core.config import load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_load_config(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "env: point_mass_maze\n"
        "algo: onestep_fb\n"
        "mode: offline\n"
        "dataset: exorl_rnd\n"
        "seeds: [0, 1]\n"
        "overrides: {agent.latent_dim: 50}\n"
    )

    config = load_config(path)

    assert config.algo == "onestep_fb"
    assert config.obs_type == "state"
    assert config.seeds == (0, 1)
    assert config.overrides == {"agent.latent_dim": 50}


def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("env: point_mass_maze\nalgo: onestep_fb\nmode: offline\nlearning_rate: 0.1\n")

    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_load_config_requires_env_algo_mode(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("env: point_mass_maze\n")

    with pytest.raises(ValueError, match="missing required keys"):
        load_config(path)


@pytest.mark.parametrize("path", sorted(CONFIGS.rglob("*.yaml")), ids=lambda p: str(p.relative_to(CONFIGS)))
def test_shipped_configs_are_compatible(path):
    from predictive_representations_rl.core import registry
    from predictive_representations_rl.core.compat import check

    config = load_config(path)
    issues = check(config, registry.get_env(config.env), registry.get_algorithm(config.algo), CONFIGS.parent)

    # Datasets and runtimes may not be on this machine; the config itself must be valid.
    assert [issue for issue in issues if issue.kind == "incompatible"] == []


@pytest.mark.parametrize("variant", ["", "../x", "a/b", ".hidden", 3])
def test_variant_must_be_a_plain_name(tmp_path, variant):
    path = tmp_path / "exp.yaml"
    path.write_text(f"env: point_mass_maze\nalgo: onestep_fb\nmode: offline\nvariant: {variant!r}\n")

    with pytest.raises(ValueError, match="variant"):
        load_config(path)


def test_extends_merges_option_blocks_and_replaces_the_rest(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "env: point_mass_maze\nalgo: onestep_fb\nmode: offline\ndataset: exorl_rnd\n"
        "budget: {train_steps: 1000}\noverrides: {agent.alpha: 0, agent.latent_dim: 50}\n"
        "resources: {queue: gpua100, walltime: '24:00'}\n"
    )
    (tmp_path / "sweep").mkdir()
    (tmp_path / "sweep" / "a.yaml").write_text(
        "extends: ../base.yaml\nvariant: alpha0.3\noverrides: {agent.alpha: 0.3}\nresources: {queue: gpul40s}\n"
    )

    config = load_config(tmp_path / "sweep" / "a.yaml")

    assert config.variant == "alpha0.3" and config.dataset == "exorl_rnd"
    assert config.overrides == {"agent.alpha": 0.3, "agent.latent_dim": 50}
    assert config.resources == {"queue": "gpul40s", "walltime": "24:00"}
    assert config.budget == {"train_steps": 1000}


def test_circular_extends_is_an_error(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")

    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")
