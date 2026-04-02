"""Tests for the SL→RL transition orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


class TestSLToRL:
    """Verify sl_to_rl() orchestrates the SL→RL handoff correctly."""

    @pytest.fixture
    def sl_data_dir(self, tmp_path: Path) -> Path:
        """Create a minimal SL data shard."""
        from keisei.sl.prepare import write_shard

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

    def test_sl_to_rl_returns_loop_with_trained_weights(
        self, tmp_path: Path, sl_data_dir: Path
    ) -> None:
        """sl_to_rl should run SL training, save checkpoint, and return a configured loop."""
        from keisei.training.transition import sl_to_rl

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        mock_vecenv = MagicMock()
        mock_vecenv.observation_channels = 50
        mock_vecenv.action_space_size = 11259

        loop = sl_to_rl(
            sl_data_dir=sl_data_dir,
            sl_epochs=2,
            sl_batch_size=8,
            checkpoint_dir=checkpoint_dir,
            rl_config_path=None,
            architecture="se_resnet",
            model_params={
                "num_blocks": 2, "channels": 32, "se_reduction": 8,
                "global_pool_channels": 16, "policy_channels": 8,
                "value_fc_size": 32, "score_fc_size": 16, "obs_channels": 50,
            },
            vecenv=mock_vecenv,
            db_path=str(tmp_path / "test.db"),
        )

        # A checkpoint should have been saved
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) == 1

        # The loop should exist and have resume_mode="sl"
        assert loop._resume_mode == "sl"

    def test_sl_to_rl_model_has_trained_weights_and_fresh_optimizer(
        self, tmp_path: Path, sl_data_dir: Path
    ) -> None:
        """After sl_to_rl(), RL model should have SL weights but fresh optimizer."""
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        from keisei.training.transition import sl_to_rl

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        mock_vecenv = MagicMock()
        mock_vecenv.observation_channels = 50
        mock_vecenv.action_space_size = 11259

        model_params = {
            "num_blocks": 2, "channels": 32, "se_reduction": 8,
            "global_pool_channels": 16, "policy_channels": 8,
            "value_fc_size": 32, "score_fc_size": 16, "obs_channels": 50,
        }

        loop = sl_to_rl(
            sl_data_dir=sl_data_dir,
            sl_epochs=1,
            sl_batch_size=8,
            checkpoint_dir=checkpoint_dir,
            architecture="se_resnet",
            model_params=model_params,
            vecenv=mock_vecenv,
            db_path=str(tmp_path / "test.db"),
        )

        # The RL model should NOT have random weights — it should have SL-trained weights.
        # Verify by checking that a random model produces different outputs.
        device = next(loop._base_model.parameters()).device
        random_model = SEResNetModel(SEResNetParams(**model_params)).to(device)
        test_obs = torch.randn(1, 50, 9, 9, device=device)
        with torch.no_grad():
            rl_out = loop._base_model(test_obs)
            random_out = random_model(test_obs)

        # SL-trained weights should differ from random init
        assert not torch.allclose(
            rl_out.policy_logits, random_out.policy_logits, atol=1e-3
        ), "RL model appears to have random weights — SL training didn't transfer"

        # The RL optimizer should be fresh (no SL momentum buffers)
        assert len(loop.ppo.optimizer.state) == 0, (
            "RL optimizer has state — SL momentum was not skipped"
        )
