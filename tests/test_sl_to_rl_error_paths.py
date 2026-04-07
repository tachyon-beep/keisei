"""Gap-analysis tests for keisei.training.transition — error paths.

Covers: SL training failure propagation, sl_epochs=0 path,
checkpoint_dir creation failure.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from keisei.db import init_db, read_training_state

pytestmark = pytest.mark.integration


@pytest.fixture
def sl_data_dir(tmp_path: Path) -> Path:
    """Create a minimal SL data shard."""
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


@pytest.fixture
def model_params() -> dict[str, int]:
    return {
        "num_blocks": 2, "channels": 32, "se_reduction": 8,
        "global_pool_channels": 16, "policy_channels": 8,
        "value_fc_size": 32, "score_fc_size": 16, "obs_channels": 50,
    }


@pytest.fixture
def mock_vecenv() -> MagicMock:
    v = MagicMock()
    v.observation_channels = 50
    v.action_space_size = 11259
    return v


class TestSLTrainingFailure:
    """Verify that SL training failures propagate cleanly."""

    def test_sl_failure_raises_and_no_db_state(
        self, tmp_path: Path, sl_data_dir: Path, model_params, mock_vecenv,
    ) -> None:
        """If SL training crashes, the exception should propagate and
        no training_state should be written to the DB."""
        from keisei.training.transition import sl_to_rl

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        db_path = str(tmp_path / "test.db")

        with patch(
            "keisei.training.transition.SLTrainer.train_epoch",
            side_effect=RuntimeError("SL training exploded"),
        ):
            with pytest.raises(RuntimeError, match="SL training exploded"):
                sl_to_rl(
                    sl_data_dir=sl_data_dir,
                    sl_epochs=2,
                    sl_batch_size=8,
                    checkpoint_dir=checkpoint_dir,
                    architecture="se_resnet",
                    model_params=model_params,
                    vecenv=mock_vecenv,
                    db_path=db_path,
                )

        # DB should not have been initialized or written to
        if Path(db_path).exists():
            init_db(db_path)
            assert read_training_state(db_path) is None

    def test_sl_failure_on_second_epoch(
        self, tmp_path: Path, sl_data_dir: Path, model_params, mock_vecenv,
    ) -> None:
        """If SL training fails on the second epoch (after one success),
        the exception should still propagate — no partial checkpoint."""
        from keisei.training.transition import sl_to_rl

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        db_path = str(tmp_path / "test.db")

        call_count = 0

        def fail_on_second(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise RuntimeError("Epoch 2 failed")
            return {"loss": 0.5}

        with patch(
            "keisei.training.transition.SLTrainer.train_epoch",
            side_effect=fail_on_second,
        ):
            with pytest.raises(RuntimeError, match="Epoch 2 failed"):
                sl_to_rl(
                    sl_data_dir=sl_data_dir,
                    sl_epochs=3,
                    sl_batch_size=8,
                    checkpoint_dir=checkpoint_dir,
                    architecture="se_resnet",
                    model_params=model_params,
                    vecenv=mock_vecenv,
                    db_path=db_path,
                )

        # No checkpoint should be saved (failure was before Phase 2)
        assert list(checkpoint_dir.glob("*.pt")) == []


class TestSLEpochsZero:
    """Test the sl_epochs=0 path (skip SL, go straight to RL)."""

    def test_zero_epochs_saves_untrained_checkpoint(
        self, tmp_path: Path, sl_data_dir: Path, model_params, mock_vecenv,
    ) -> None:
        """sl_epochs=0 should save a checkpoint with untrained weights
        and return a valid RL loop."""
        from keisei.training.transition import sl_to_rl

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        db_path = str(tmp_path / "test.db")

        loop = sl_to_rl(
            sl_data_dir=sl_data_dir,
            sl_epochs=0,
            sl_batch_size=8,
            checkpoint_dir=checkpoint_dir,
            architecture="se_resnet",
            model_params=model_params,
            vecenv=mock_vecenv,
            db_path=db_path,
        )

        # Checkpoint should exist
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) == 1

        # DB state should be written
        state = read_training_state(db_path)
        assert state is not None
        assert state["current_epoch"] == 0
        assert state["checkpoint_path"] == str(checkpoints[0])

        # Loop should be configured for SL resume
        assert loop._resume_mode == "sl"

    def test_zero_epochs_no_train_epoch_called(
        self, tmp_path: Path, sl_data_dir: Path, model_params, mock_vecenv,
    ) -> None:
        """sl_epochs=0 should never call SLTrainer.train_epoch."""
        from keisei.training.transition import sl_to_rl

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        with patch(
            "keisei.training.transition.SLTrainer.train_epoch",
        ) as mock_train:
            sl_to_rl(
                sl_data_dir=sl_data_dir,
                sl_epochs=0,
                sl_batch_size=8,
                checkpoint_dir=checkpoint_dir,
                architecture="se_resnet",
                model_params=model_params,
                vecenv=mock_vecenv,
                db_path=str(tmp_path / "test.db"),
            )
            mock_train.assert_not_called()


class TestArchitectureMismatch:
    """sl_to_rl must reject mismatched SL args vs rl_config_path model config."""

    def test_architecture_mismatch_raises(
        self, tmp_path: Path, sl_data_dir: Path, model_params, mock_vecenv,
    ) -> None:
        """If SL architecture differs from rl_config model architecture, raise immediately."""
        from keisei.training.transition import sl_to_rl

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # RL config specifies "transformer" but SL args say "se_resnet"
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
db_path = "{tmp_path / 'test.db'}"

[model]
display_name = "TestBot"
architecture = "transformer"

[model.params]
d_model = 32
nhead = 4
num_layers = 2
"""
        )

        with pytest.raises(ValueError, match="architecture.*mismatch"):
            sl_to_rl(
                sl_data_dir=sl_data_dir,
                sl_epochs=1,
                sl_batch_size=8,
                checkpoint_dir=checkpoint_dir,
                rl_config_path=rl_config_path,
                architecture="se_resnet",
                model_params=model_params,
                vecenv=mock_vecenv,
                db_path=str(tmp_path / "test.db"),
            )

        # No checkpoint should exist — we failed before SL training
        assert list(checkpoint_dir.glob("*.pt")) == []

    def test_param_shape_mismatch_raises(
        self, tmp_path: Path, sl_data_dir: Path, model_params, mock_vecenv,
    ) -> None:
        """Same architecture name but different model params must raise before SL training."""
        from keisei.training.transition import sl_to_rl

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # RL config uses se_resnet but with different channels/blocks than SL
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
db_path = "{tmp_path / 'test.db'}"

[model]
display_name = "TestBot"
architecture = "se_resnet"

[model.params]
num_blocks = 10
channels = 128
se_reduction = 16
global_pool_channels = 64
policy_channels = 16
value_fc_size = 128
score_fc_size = 64
obs_channels = 50
"""
        )

        with pytest.raises(ValueError, match="param.*mismatch"):
            sl_to_rl(
                sl_data_dir=sl_data_dir,
                sl_epochs=1,
                sl_batch_size=8,
                checkpoint_dir=checkpoint_dir,
                rl_config_path=rl_config_path,
                architecture="se_resnet",
                model_params=model_params,
                vecenv=mock_vecenv,
                db_path=str(tmp_path / "test.db"),
            )

        # No checkpoint should exist — we failed before SL training
        assert list(checkpoint_dir.glob("*.pt")) == []


class TestCheckpointDirCreation:
    """Test checkpoint_dir auto-creation and failure handling."""

    def test_checkpoint_dir_created_automatically(
        self, tmp_path: Path, sl_data_dir: Path, model_params, mock_vecenv,
    ) -> None:
        """sl_to_rl should create checkpoint_dir if it doesn't exist."""
        from keisei.training.transition import sl_to_rl

        checkpoint_dir = tmp_path / "deep" / "nested" / "checkpoints"
        assert not checkpoint_dir.exists()

        sl_to_rl(
            sl_data_dir=sl_data_dir,
            sl_epochs=0,
            sl_batch_size=8,
            checkpoint_dir=checkpoint_dir,
            architecture="se_resnet",
            model_params=model_params,
            vecenv=mock_vecenv,
            db_path=str(tmp_path / "test.db"),
        )

        assert checkpoint_dir.exists()
        assert len(list(checkpoint_dir.glob("*.pt"))) == 1

    def test_readonly_checkpoint_dir_raises(
        self, tmp_path: Path, sl_data_dir: Path, model_params, mock_vecenv,
    ) -> None:
        """If checkpoint_dir can't be created, error should propagate."""
        import os

        if os.getuid() == 0:
            pytest.skip("cannot test permission errors as root")

        from keisei.training.transition import sl_to_rl

        # Use a path under a read-only parent
        readonly_parent = tmp_path / "readonly"
        readonly_parent.mkdir()
        readonly_parent.chmod(0o444)

        checkpoint_dir = readonly_parent / "checkpoints"

        try:
            with pytest.raises((PermissionError, OSError)):
                sl_to_rl(
                    sl_data_dir=sl_data_dir,
                    sl_epochs=0,
                    sl_batch_size=8,
                    checkpoint_dir=checkpoint_dir,
                    architecture="se_resnet",
                    model_params=model_params,
                    vecenv=mock_vecenv,
                    db_path=str(tmp_path / "test.db"),
                )
        finally:
            # Restore permissions for cleanup
            readonly_parent.chmod(0o755)
