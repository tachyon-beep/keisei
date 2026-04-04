# tests/test_katago_config.py
"""Tests for KataGo config extensions."""

from pathlib import Path

import pytest

from keisei.config import DistributedConfig, load_config

KATAGO_TOML = """
[model]
display_name = "KataGo-SE-b2c32"
architecture = "se_resnet"

[model.params]
num_blocks = 2
channels = 32
se_reduction = 8
global_pool_channels = 16
policy_channels = 8
value_fc_size = 32
score_fc_size = 16
obs_channels = 50

[training]
algorithm = "katago_ppo"
num_games = 2
max_ply = 50
checkpoint_interval = 10
checkpoint_dir = "checkpoints/"

[training.algorithm_params]
learning_rate = 0.0002
gamma = 0.99
lambda_policy = 1.0
lambda_value = 1.5
lambda_score = 0.02
lambda_entropy = 0.01
score_normalization = 76.0
grad_clip = 1.0

[display]
moves_per_minute = 0
db_path = "test.db"
"""


def test_load_katago_config(tmp_path):
    toml_file = tmp_path / "katago.toml"
    toml_file.write_text(KATAGO_TOML)
    config = load_config(toml_file)
    assert config.model.architecture == "se_resnet"
    assert config.training.algorithm == "katago_ppo"
    assert config.model.params["num_blocks"] == 2
    assert config.model.params["channels"] == 32


def test_invalid_architecture_rejected(tmp_path):
    toml = KATAGO_TOML.replace('architecture = "se_resnet"', 'architecture = "invalid"')
    toml_file = tmp_path / "bad.toml"
    toml_file.write_text(toml)
    with pytest.raises(ValueError, match="Unknown architecture"):
        load_config(toml_file)


def test_load_real_katago_toml():
    """Verify the shipped keisei-katago.toml parses without error."""
    toml_path = Path(__file__).parent.parent / "keisei-katago.toml"
    if not toml_path.exists():
        pytest.skip("keisei-katago.toml not present")
    config = load_config(toml_path)
    assert config.model.architecture == "se_resnet"
    assert config.training.algorithm == "katago_ppo"


def test_invalid_algorithm_rejected(tmp_path):
    """Unknown algorithm should raise ValueError."""
    toml = KATAGO_TOML.replace('algorithm = "katago_ppo"', 'algorithm = "bad_algo"')
    toml_file = tmp_path / "bad.toml"
    toml_file.write_text(toml)
    with pytest.raises(ValueError, match="Unknown algorithm"):
        load_config(toml_file)



class TestDistributedConfig:
    def test_defaults(self):
        cfg = DistributedConfig()
        assert cfg.sync_batchnorm is True
        assert cfg.find_unused_parameters is False
        assert cfg.gradient_as_bucket_view is True

    def test_custom_values(self):
        cfg = DistributedConfig(sync_batchnorm=False)
        assert cfg.sync_batchnorm is False

    def test_rejects_unknown_keys(self):
        """Typos in config keys should fail loudly."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            DistributedConfig(sycn_batchnorm=True)
