"""Tests for transition.py warning log on db_path mismatch.

GAP-M1: The warning at transition.py lines 144-152 is not exercised
by existing tests.  This test captures the warning log and verifies
the message includes both paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.integration


class TestTransitionDbPathWarning:
    """Verify the warning log when db_path differs from rl_config.display.db_path."""

    @pytest.fixture
    def sl_data_dir(self, tmp_path: Path) -> Path:
        from keisei.sl.dataset import write_shard

        data_dir = tmp_path / "sl_data"
        data_dir.mkdir()
        n = 16
        rng = np.random.default_rng(42)
        write_shard(
            data_dir / "shard_000.bin",
            rng.standard_normal((n, 50 * 81)).astype(np.float32),
            rng.integers(0, 11259, size=n).astype(np.int64),
            rng.integers(0, 3, size=n).astype(np.int64),
            rng.standard_normal(n).astype(np.float32),
        )
        return data_dir

    def test_warning_logged_when_db_paths_differ(
        self, tmp_path: Path, sl_data_dir: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When db_path arg != rl_config.display.db_path, a warning should be logged."""
        from keisei.training.transition import sl_to_rl

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        db_arg = str(tmp_path / "arg.db")
        db_cfg = tmp_path / "cfg.db"

        rl_config_path = tmp_path / "rl.toml"
        rl_config_path.write_text(
            f"""\
[training]
algorithm = "katago_ppo"
num_games = 1
max_ply = 20
checkpoint_interval = 10
checkpoint_dir = "{checkpoint_dir}"

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
db_path = "{db_cfg}"

[model]
display_name = "TestBot"
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
"""
        )

        mock_vecenv = MagicMock()
        mock_vecenv.observation_channels = 50
        mock_vecenv.action_space_size = 11259

        with caplog.at_level(logging.WARNING, logger="keisei.training.transition"):
            sl_to_rl(
                sl_data_dir=sl_data_dir,
                sl_epochs=0,
                sl_batch_size=8,
                checkpoint_dir=checkpoint_dir,
                rl_config_path=rl_config_path,
                architecture="se_resnet",
                model_params={
                    "num_blocks": 2, "channels": 32, "se_reduction": 8,
                    "global_pool_channels": 16, "policy_channels": 8,
                    "value_fc_size": 32, "score_fc_size": 16, "obs_channels": 50,
                },
                vecenv=mock_vecenv,
                db_path=db_arg,
            )

        # Find the warning about mismatched db_path
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) >= 1, "Expected at least one WARNING log"

        # The warning message should mention both paths
        warning_text = " ".join(r.message for r in warnings)
        assert "db_path" in warning_text.lower() or "differs" in warning_text.lower(), (
            f"Warning should mention db_path mismatch, got: {warning_text}"
        )
        assert str(db_cfg) in warning_text, (
            f"Warning should mention rl_config db_path ({db_cfg}), got: {warning_text}"
        )

    def test_no_warning_when_db_paths_match(
        self, tmp_path: Path, sl_data_dir: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When db_path matches rl_config.display.db_path, no warning should be logged."""
        from keisei.training.transition import sl_to_rl

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        db_path = str(tmp_path / "same.db")

        rl_config_path = tmp_path / "rl.toml"
        rl_config_path.write_text(
            f"""\
[training]
algorithm = "katago_ppo"
num_games = 1
max_ply = 20
checkpoint_interval = 10
checkpoint_dir = "{checkpoint_dir}"

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
db_path = "{db_path}"

[model]
display_name = "TestBot"
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
"""
        )

        mock_vecenv = MagicMock()
        mock_vecenv.observation_channels = 50
        mock_vecenv.action_space_size = 11259

        with caplog.at_level(logging.WARNING, logger="keisei.training.transition"):
            sl_to_rl(
                sl_data_dir=sl_data_dir,
                sl_epochs=0,
                sl_batch_size=8,
                checkpoint_dir=checkpoint_dir,
                rl_config_path=rl_config_path,
                architecture="se_resnet",
                model_params={
                    "num_blocks": 2, "channels": 32, "se_reduction": 8,
                    "global_pool_channels": 16, "policy_channels": 8,
                    "value_fc_size": 32, "score_fc_size": 16, "obs_channels": 50,
                },
                vecenv=mock_vecenv,
                db_path=db_path,
            )

        # No "differs" warning should appear
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "differs" in r.message
        ]
        assert len(warnings) == 0, (
            f"No db_path mismatch warning expected, got: {[r.message for r in warnings]}"
        )
