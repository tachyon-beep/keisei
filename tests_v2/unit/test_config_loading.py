"""
Tests for YAML config loading and CLI override parsing.

Validates that YAML files are correctly loaded, merged with defaults,
and converted into AppConfig instances via keisei.utils.utils.load_config.
"""

import pytest
import yaml

from keisei.config_schema import AppConfig, EnvConfig, TrainingConfig
from keisei.utils.utils import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path, data, filename="test_config.yaml"):
    """Write a dict as YAML to a temp file and return the path string."""
    yaml_path = tmp_path / filename
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    return str(yaml_path)


def _minimal_config_data(**training_overrides):
    """Return a minimal config dict with optional training overrides."""
    data = {
        "env": {"device": "cpu", "seed": 123},
        "training": {
            "learning_rate": 0.001,
            "total_timesteps": 100,
            "steps_per_epoch": 16,
            "minibatch_size": 8,
        },
        "evaluation": {"num_games": 5},
        "logging": {"log_file": "test.log"},
        "wandb": {"enabled": False},
        "parallel": {"enabled": False},
    }
    data["training"].update(training_overrides)
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadFromYaml:
    def test_load_from_valid_yaml_creates_app_config(self, tmp_path):
        """A well-formed YAML file should load into a valid AppConfig."""
        config_data = _minimal_config_data()
        yaml_path = _write_yaml(tmp_path, config_data)

        config = load_config(yaml_path)
        assert isinstance(config, AppConfig)
        assert config.training.learning_rate == 0.001

    def test_default_config_file_loads_successfully(self):
        """The project default_config.yaml must always load without errors."""
        config = load_config()  # None path loads default_config.yaml
        assert isinstance(config, AppConfig)
        assert config.training.learning_rate > 0

    def test_yaml_with_overridden_learning_rate(self, tmp_path):
        """An overridden learning_rate in YAML should propagate correctly."""
        config_data = _minimal_config_data(learning_rate=0.05)
        yaml_path = _write_yaml(tmp_path, config_data)

        config = load_config(yaml_path)
        assert config.training.learning_rate == 0.05

    def test_yaml_with_partial_sections_uses_defaults(self, tmp_path):
        """A YAML file with only some sections should fill in defaults."""
        partial_data = {
            "training": {"learning_rate": 0.007, "minibatch_size": 4},
        }
        yaml_path = _write_yaml(tmp_path, partial_data)

        config = load_config(yaml_path)
        assert config.training.learning_rate == 0.007
        # Other sections should fall back to defaults from default_config.yaml
        assert isinstance(config.env, EnvConfig)

    def test_cli_override_applies_correctly(self, tmp_path):
        """CLI overrides should take precedence over YAML values."""
        config_data = _minimal_config_data(learning_rate=0.001)
        yaml_path = _write_yaml(tmp_path, config_data)

        config = load_config(
            yaml_path,
            cli_overrides={"training.learning_rate": 0.999},
        )
        assert config.training.learning_rate == 0.999


class TestYamlRoundtrip:
    def test_yaml_roundtrip_preserves_values(self, tmp_path):
        """Dump config to YAML, reload, and verify values match."""
        original = load_config()
        dumped = original.model_dump()
        yaml_path = _write_yaml(tmp_path, dumped, filename="roundtrip.yaml")

        reloaded = load_config(yaml_path)
        assert reloaded.training.learning_rate == original.training.learning_rate
        assert reloaded.env.device == original.env.device
        assert reloaded.evaluation.num_games == original.evaluation.num_games


class TestYamlEdgeCases:
    def test_empty_yaml_uses_all_defaults(self, tmp_path):
        """An empty YAML file should fall back to all defaults."""
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")

        config = load_config(str(yaml_path))
        assert isinstance(config, AppConfig)
        # Defaults from default_config.yaml should be used
        assert config.training.learning_rate > 0

    def test_invalid_yaml_raises_error(self, tmp_path):
        """A YAML file with invalid syntax should raise an error."""
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("{{{{not valid yaml: ][}")

        with pytest.raises(Exception):
            load_config(str(yaml_path))
