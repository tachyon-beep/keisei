# tests/test_katago_ppo.py
"""Tests for the KataGo multi-head PPO algorithm."""

import pytest
import torch

from keisei.training.katago_ppo import (
    KataGoPPOAlgorithm,
    KataGoPPOParams,
    KataGoRolloutBuffer,
    compute_value_metrics,
)
from keisei.training.gae import compute_gae
from keisei.training.algorithm_registry import validate_algorithm_params, VALID_ALGORITHMS
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


class TestKataGoPPOParams:
    def test_defaults(self):
        params = KataGoPPOParams()
        assert params.learning_rate == 2e-4
        assert params.gamma == 0.99
        assert params.gae_lambda == 0.95
        assert params.lambda_value == 1.5
        assert params.lambda_score == 0.02
        assert params.lambda_entropy == 0.01
        assert params.score_normalization == 76.0
        assert params.grad_clip == 1.0

    def test_custom_params(self):
        params = KataGoPPOParams(learning_rate=1e-3, gamma=1.0)
        assert params.learning_rate == 1e-3
        assert params.gamma == 1.0


class TestKataGoRolloutBuffer:
    def test_add_and_size(self):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        obs = torch.randn(2, 50, 9, 9)
        actions = torch.tensor([100, 200])
        log_probs = torch.tensor([0.5, 0.6])
        values = torch.tensor([0.1, 0.2])
        rewards = torch.tensor([0.0, 0.0])
        dones = torch.tensor([False, False])
        legal_masks = torch.ones(2, 11259, dtype=torch.bool)
        value_cats = torch.tensor([0, 2])  # W, L
        score_targets = torch.tensor([0.5, -0.3])

        buf.add(obs, actions, log_probs, values, rewards, dones, legal_masks,
                value_cats, score_targets)
        assert buf.size == 1

    def test_flatten(self):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(3):
            buf.add(
                torch.randn(2, 50, 9, 9),
                torch.randint(0, 11259, (2,)),
                torch.randn(2),
                torch.randn(2),
                torch.randn(2),
                torch.zeros(2, dtype=torch.bool),
                torch.ones(2, 11259, dtype=torch.bool),
                torch.randint(0, 3, (2,)),
                torch.rand(2) * 2 - 1,  # uniform in [-1, 1] to stay within guard
            )
        data = buf.flatten()
        assert data["observations"].shape == (6, 50, 9, 9)
        assert data["actions"].shape == (6,)
        assert data["value_categories"].shape == (6,)
        assert data["score_targets"].shape == (6,)

    def test_clear(self):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        buf.add(
            torch.randn(2, 50, 9, 9), torch.zeros(2, dtype=torch.long),
            torch.zeros(2), torch.zeros(2), torch.zeros(2),
            torch.zeros(2, dtype=torch.bool), torch.ones(2, 11259, dtype=torch.bool),
            torch.zeros(2, dtype=torch.long), torch.zeros(2),
        )
        assert buf.size == 1
        buf.clear()
        assert buf.size == 0

    def test_unnormalized_score_targets_rejected(self):
        """Buffer should reject score_targets that appear unnormalized (abs > 2.0)."""
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        with pytest.raises(ValueError, match="unnormalized"):
            buf.add(
                torch.randn(2, 50, 9, 9), torch.zeros(2, dtype=torch.long),
                torch.zeros(2), torch.zeros(2), torch.zeros(2),
                torch.zeros(2, dtype=torch.bool), torch.ones(2, 11259, dtype=torch.bool),
                torch.zeros(2, dtype=torch.long), torch.tensor([10.0, -5.0]),
            )


class TestAlgorithmRegistry:
    def test_katago_ppo_in_valid_algorithms(self):
        assert "katago_ppo" in VALID_ALGORITHMS

    def test_validate_katago_ppo_params(self):
        validated = validate_algorithm_params("katago_ppo", {"learning_rate": 1e-3})
        assert isinstance(validated, KataGoPPOParams)
        assert validated.learning_rate == 1e-3


@pytest.fixture
def small_model():
    params = SEResNetParams(
        num_blocks=2, channels=32, se_reduction=8,
        global_pool_channels=16, policy_channels=8,
        value_fc_size=32, score_fc_size=16, obs_channels=50,
    )
    return SEResNetModel(params)


@pytest.fixture
def ppo(small_model):
    params = KataGoPPOParams()
    return KataGoPPOAlgorithm(params, small_model)


class TestKataGoPPOActionSelection:
    def test_select_actions_shapes(self, ppo):
        obs = torch.randn(4, 50, 9, 9)
        legal_masks = torch.ones(4, 11259, dtype=torch.bool)
        actions, log_probs, values = ppo.select_actions(obs, legal_masks)
        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)

    def test_select_actions_values_bounded(self, ppo):
        """Scalar value P(W) - P(L) should be in [-1, 1]."""
        obs = torch.randn(8, 50, 9, 9)
        legal_masks = torch.ones(8, 11259, dtype=torch.bool)
        _, _, values = ppo.select_actions(obs, legal_masks)
        assert values.min() >= -1.0
        assert values.max() <= 1.0

    def test_select_actions_all_illegal_raises(self, ppo):
        """All-False legal mask should raise, not produce NaN."""
        obs = torch.randn(2, 50, 9, 9)
        legal_masks = torch.zeros(2, 11259, dtype=torch.bool)  # all illegal
        with pytest.raises(RuntimeError, match="zero legal actions"):
            ppo.select_actions(obs, legal_masks)

    def test_select_actions_respects_mask(self, ppo):
        """Actions should only be sampled from legal positions (20 trials)."""
        obs = torch.randn(2, 50, 9, 9)
        legal_masks = torch.zeros(2, 11259, dtype=torch.bool)
        legal_masks[:, 0] = True
        legal_masks[:, 1000] = True
        for _ in range(20):
            actions, _, _ = ppo.select_actions(obs, legal_masks)
            for a in actions.tolist():
                assert a in (0, 1000), f"Action {a} should be 0 or 1000"


class TestValueMetrics:
    def test_compute_value_metrics(self):
        """Verify the metrics helper computes frac_predicted_win/draw/loss correctly."""
        # 4 predictions: 2 predict W, 1 predicts D, 1 predicts L
        value_logits = torch.tensor([
            [2.0, 0.0, 0.0],  # predicts W
            [0.0, 2.0, 0.0],  # predicts D
            [0.0, 0.0, 2.0],  # predicts L
            [1.5, 0.0, 0.0],  # predicts W
        ])
        value_targets = torch.tensor([0, 1, 2, 0])  # W, D, L, W

        metrics = compute_value_metrics(value_logits, value_targets)
        assert metrics["value_accuracy"] == 1.0  # all correct
        assert metrics["frac_predicted_win"] == 0.5
        assert metrics["frac_predicted_draw"] == 0.25
        assert metrics["frac_predicted_loss"] == 0.25


class TestKataGoPPOUpdate:
    def test_update_returns_metrics(self, ppo):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(
                obs, actions, log_probs, values,
                torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                torch.randint(0, 3, (2,)), torch.rand(2) * 2 - 1,
            )
        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "score_loss" in losses
        assert "entropy" in losses
        assert "gradient_norm" in losses

    def test_update_loss_values_are_finite(self, ppo):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(
                obs, actions, log_probs, values,
                torch.randn(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                torch.randint(0, 3, (2,)), torch.rand(2) * 2 - 1,
            )
        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        for key, val in losses.items():
            assert not torch.tensor(val).isnan(), f"{key} is NaN"
            assert not torch.tensor(val).isinf(), f"{key} is inf"

    def test_update_clears_buffer(self, ppo):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(
                obs, actions, log_probs, values,
                torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                torch.zeros(2, dtype=torch.long), torch.zeros(2),
            )
        ppo.update(buf, torch.zeros(2))
        assert buf.size == 0

    def test_update_changes_model_parameters(self, ppo):
        """Optimizer step should actually modify model weights."""
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(
                obs, actions, log_probs, values,
                torch.randn(2) * 0.1, torch.zeros(2, dtype=torch.bool), legal_masks,
                torch.randint(0, 3, (2,)), torch.rand(2) * 2 - 1,
            )
        params_before = {
            n: p.clone() for n, p in ppo.model.named_parameters()
        }
        ppo.update(buf, torch.zeros(2))
        changed = any(
            not torch.equal(params_before[n], p)
            for n, p in ppo.model.named_parameters()
        )
        assert changed, "update() should modify at least some model parameters"

    def test_update_returns_value_prediction_metrics(self, ppo):
        """update() should include value_accuracy and frac_predicted_* in metrics."""
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(
                obs, actions, log_probs, values,
                torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                torch.randint(0, 3, (2,)), torch.rand(2) * 2 - 1,
            )
        losses = ppo.update(buf, torch.zeros(2))
        assert "value_accuracy" in losses, "Should include value prediction metrics"
        assert "frac_predicted_win" in losses
        assert "frac_predicted_draw" in losses
        assert "frac_predicted_loss" in losses


class TestIgnoreIndex:
    """Test the ignore_index=-1 path for non-terminal value categories."""

    def test_update_with_ignore_index_values(self, ppo):
        """Passing value_categories=-1 should not crash and should produce finite loss."""
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            # All value_cats = -1 (non-terminal, should be ignored by cross_entropy)
            buf.add(
                obs, actions, log_probs, values,
                torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                torch.full((2,), -1, dtype=torch.long), torch.zeros(2),
            )
        losses = ppo.update(buf, torch.zeros(2))
        for key, val in losses.items():
            if key.startswith("frac_") or key == "value_accuracy":
                continue  # metrics may not be present if all cats are -1
            assert not torch.tensor(val).isnan(), f"{key} is NaN with all-ignore value_cats"


class TestGAEUnit:
    """Isolated unit tests for compute_gae — multi-step accumulation and episode boundaries."""

    def test_multi_step_accumulation(self):
        """Verify GAE recursive accumulation with hand-computed reference values."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        dones = torch.zeros(3, dtype=torch.bool)
        next_value = torch.tensor(0.0)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)

        # Step 2: delta = 3.0 + 0.99*0.0 - 0.5 = 2.5, gae = 2.5
        assert abs(advantages[2].item() - 2.5) < 1e-3
        # Step 1: delta = 2.0 + 0.99*0.5 - 0.5 = 1.995, gae = 1.995 + 0.99*0.95*2.5 = 4.346
        assert abs(advantages[1].item() - 4.346) < 1e-2
        # Step 0: delta = 1.0 + 0.99*0.5 - 0.5 = 0.995, gae = 0.995 + 0.99*0.95*4.346 ≈ 5.08
        assert abs(advantages[0].item() - 5.08) < 0.05

    def test_episode_boundary_resets_gae(self):
        """done=True at step 1 should zero out advantage propagation."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        dones = torch.tensor([False, True, False])
        next_value = torch.tensor(0.0)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)

        # Step 1 is terminal: delta = 2.0 + 0 - 0.5 = 1.5, gae resets
        assert abs(advantages[1].item() - 1.5) < 1e-3
        # Step 0: delta = 1.0 + 0.99*0.5*(1-0) - 0.5 = 0.995
        # gae = 0.995 + 0.99*0.95*(1-0)*1.5 = 0.995 + 1.41075 = 2.406
        assert abs(advantages[0].item() - 2.406) < 0.01

    def test_single_step(self):
        """Single-step trajectory should just be the TD residual."""
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5])
        dones = torch.tensor([False])
        next_value = torch.tensor(0.3)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        # delta = 1.0 + 0.99*0.3 - 0.5 = 0.797
        assert abs(advantages[0].item() - 0.797) < 1e-3


class TestBufferEdgeCases:
    def test_empty_buffer_flatten_raises(self):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        with pytest.raises(ValueError, match="Cannot flatten an empty buffer"):
            buf.flatten()

    def test_invalid_value_categories_rejected(self):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        with pytest.raises(ValueError, match="invalid values"):
            buf.add(
                torch.randn(2, 50, 9, 9), torch.zeros(2, dtype=torch.long),
                torch.zeros(2), torch.zeros(2), torch.zeros(2),
                torch.zeros(2, dtype=torch.bool), torch.ones(2, 11259, dtype=torch.bool),
                torch.tensor([5, 3], dtype=torch.long),  # invalid categories
                torch.zeros(2),
            )


class TestScoreLossNoNaN:
    """Tests verifying NaN masking is removed and dense score targets work."""

    @pytest.fixture
    def ppo(self):
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        model = SEResNetModel(params)
        return KataGoPPOAlgorithm(KataGoPPOParams(), model)

    def test_score_loss_computed_over_full_batch(self, ppo):
        """Score loss uses ALL samples — proves NaN masking is gone."""
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(
                obs, actions, log_probs, values,
                torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                torch.randint(0, 3, (2,)), torch.tensor([0.5, -0.3]),
            )
        losses = ppo.update(buf, torch.zeros(2))
        assert losses["score_loss"] > 0, "Score loss should be non-zero with real targets"
        assert not torch.tensor(losses["score_loss"]).isnan()

    def test_nan_score_targets_rejected(self):
        """NaN score targets should be rejected by the buffer guard."""
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        with pytest.raises(ValueError, match="NaN"):
            buf.add(
                torch.randn(2, 50, 9, 9), torch.zeros(2, dtype=torch.long),
                torch.zeros(2), torch.zeros(2), torch.zeros(2),
                torch.zeros(2, dtype=torch.bool), torch.ones(2, 11259, dtype=torch.bool),
                torch.zeros(2, dtype=torch.long),
                torch.tensor([float("nan"), 0.5]),
            )

    def test_unnormalized_score_targets_rejected(self):
        """Score targets above 3.0 threshold should be rejected."""
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        with pytest.raises(ValueError, match="unnormalized"):
            buf.add(
                torch.randn(2, 50, 9, 9), torch.zeros(2, dtype=torch.long),
                torch.zeros(2), torch.zeros(2), torch.zeros(2),
                torch.zeros(2, dtype=torch.bool), torch.ones(2, 11259, dtype=torch.bool),
                torch.zeros(2, dtype=torch.long),
                torch.tensor([10.0, -5.0]),
            )
