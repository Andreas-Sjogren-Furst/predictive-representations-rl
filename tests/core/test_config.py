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
