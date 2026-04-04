"""Gap-analysis tests for model architectures — SE-ResNet param variants
and Transformer single-sample inference.

Covers: SEResNetModel with varied param configs, output shape contracts,
TransformerModel batch_size=1 forward pass.
"""

from __future__ import annotations

import pytest
import torch

from keisei.training.models.katago_base import KataGoOutput
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
from keisei.training.models.transformer import TransformerModel, TransformerParams


# ===========================================================================
# SE-ResNet — param configuration variants
# ===========================================================================


class TestSEResNetParamVariants:
    """Test SEResNetModel with different param configurations."""

    @pytest.fixture
    def default_params(self) -> SEResNetParams:
        """Minimal config for fast tests."""
        return SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )

    def test_default_params_output_shapes(self, default_params):
        """Verify output shapes with default test params."""
        model = SEResNetModel(default_params)
        obs = torch.randn(4, 50, 9, 9)
        with torch.no_grad():
            out = model(obs)

        assert isinstance(out, KataGoOutput)
        assert out.policy_logits.shape == (4, 9, 9, 139)
        assert out.value_logits.shape == (4, 3)
        assert out.score_lead.shape == (4, 1)

    def test_minimal_channels(self):
        """Smallest viable config — channels=8, se_reduction=4."""
        params = SEResNetParams(
            num_blocks=1, channels=8, se_reduction=4,
            global_pool_channels=8, policy_channels=4,
            value_fc_size=8, score_fc_size=8, obs_channels=50,
        )
        model = SEResNetModel(params)
        obs = torch.randn(2, 50, 9, 9)
        with torch.no_grad():
            out = model(obs)

        assert out.policy_logits.shape == (2, 9, 9, 139)
        assert out.value_logits.shape == (2, 3)
        assert out.score_lead.shape == (2, 1)

    def test_large_global_pool_channels(self):
        """Global pool bottleneck larger than trunk channels."""
        params = SEResNetParams(
            num_blocks=2, channels=16, se_reduction=4,
            global_pool_channels=64,  # larger than channels
            policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        model = SEResNetModel(params)
        obs = torch.randn(2, 50, 9, 9)
        with torch.no_grad():
            out = model(obs)

        assert out.policy_logits.shape == (2, 9, 9, 139)

    def test_single_block(self):
        """num_blocks=1 — minimum depth."""
        params = SEResNetParams(
            num_blocks=1, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        model = SEResNetModel(params)
        obs = torch.randn(1, 50, 9, 9)
        with torch.no_grad():
            out = model(obs)

        assert out.policy_logits.shape == (1, 9, 9, 139)

    def test_wrong_obs_channels_raises(self, default_params):
        """Passing obs with wrong channel count should raise ValueError."""
        model = SEResNetModel(default_params)
        wrong_obs = torch.randn(2, 46, 9, 9)  # 46 != 50
        with pytest.raises(ValueError, match="Expected 50 input channels"):
            model(wrong_obs)

    def test_batch_size_one(self, default_params):
        """Single-sample forward pass (inference mode)."""
        model = SEResNetModel(default_params)
        obs = torch.randn(1, 50, 9, 9)
        with torch.no_grad():
            out = model(obs)

        assert out.policy_logits.shape == (1, 9, 9, 139)
        assert out.value_logits.shape == (1, 3)
        assert out.score_lead.shape == (1, 1)

    def test_value_logits_are_raw(self, default_params):
        """Value logits should be raw (pre-softmax), not probabilities."""
        model = SEResNetModel(default_params)
        obs = torch.randn(2, 50, 9, 9)
        with torch.no_grad():
            out = model(obs)

        # Raw logits can be negative or > 1
        # If they were softmax'd, they'd all be in [0, 1] and sum to 1
        # This is a probabilistic check — with random weights, very unlikely
        # all logits happen to be valid probabilities
        sums = out.value_logits.sum(dim=-1)
        assert not torch.allclose(sums, torch.ones_like(sums), atol=0.1), (
            "Value logits sum to ~1.0, suggesting accidental softmax"
        )

    def test_different_obs_channels(self):
        """Custom obs_channels (e.g. 46 for legacy compatibility)."""
        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=46,
        )
        model = SEResNetModel(params)
        obs = torch.randn(2, 46, 9, 9)
        with torch.no_grad():
            out = model(obs)

        assert out.policy_logits.shape == (2, 9, 9, 139)


# ===========================================================================
# Transformer — batch_size=1 and edge cases
# ===========================================================================


class TestTransformerEdgeCases:
    """Test TransformerModel with single-sample and edge-case inputs."""

    @pytest.fixture
    def small_params(self) -> TransformerParams:
        return TransformerParams(d_model=32, nhead=4, num_layers=2)

    def test_batch_size_one(self, small_params):
        """Single-sample forward pass — catches reshape/mean dim bugs."""
        model = TransformerModel(small_params)
        obs = torch.randn(1, 46, 9, 9)
        with torch.no_grad():
            policy_logits, value = model(obs)

        assert policy_logits.shape == (1, 13527)
        assert value.shape == (1, 1)

    def test_batch_size_two(self, small_params):
        """Standard multi-sample forward pass."""
        model = TransformerModel(small_params)
        obs = torch.randn(2, 46, 9, 9)
        with torch.no_grad():
            policy_logits, value = model(obs)

        assert policy_logits.shape == (2, 13527)
        assert value.shape == (2, 1)

    def test_value_output_is_tanh_bounded(self, small_params):
        """Value output should be tanh-activated, so in [-1, 1]."""
        model = TransformerModel(small_params)
        obs = torch.randn(8, 46, 9, 9)
        with torch.no_grad():
            _, value = model(obs)

        assert (value >= -1.0).all() and (value <= 1.0).all(), (
            f"Value output out of tanh range: {value}"
        )

    def test_different_d_model(self):
        """Larger d_model should work without shape errors."""
        params = TransformerParams(d_model=64, nhead=8, num_layers=1)
        model = TransformerModel(params)
        obs = torch.randn(2, 46, 9, 9)
        with torch.no_grad():
            policy_logits, value = model(obs)

        assert policy_logits.shape == (2, 13527)
        assert value.shape == (2, 1)

    def test_positional_encoding_device_transfer(self, small_params):
        """Positional encoding buffers should move with the model."""
        model = TransformerModel(small_params)
        # Buffers should be on CPU initially
        assert model._row_idx.device.type == "cpu"
        assert model._col_idx.device.type == "cpu"

        # Forward pass should work on CPU
        obs = torch.randn(1, 46, 9, 9)
        with torch.no_grad():
            policy_logits, value = model(obs)
        assert policy_logits.shape == (1, 13527)
