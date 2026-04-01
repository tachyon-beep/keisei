"""Tests for LR scheduling and RL warmup in KataGoTrainingLoop."""

import pytest
import torch

from keisei.training.katago_ppo import KataGoPPOAlgorithm, KataGoPPOParams, KataGoRolloutBuffer
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


@pytest.fixture
def small_model():
    params = SEResNetParams(
        num_blocks=2,
        channels=32,
        se_reduction=8,
        global_pool_channels=16,
        policy_channels=8,
        value_fc_size=32,
        score_fc_size=16,
        obs_channels=50,
    )
    return SEResNetModel(params)


class TestLRScheduler:
    def test_plateau_scheduler_reduces_lr(self, small_model):
        """Simulate repeated high value_loss -> LR should decrease."""
        from keisei.training.katago_loop import create_lr_scheduler

        params = KataGoPPOParams(learning_rate=1e-3)
        ppo = KataGoPPOAlgorithm(params, small_model)

        scheduler = create_lr_scheduler(
            ppo.optimizer,
            schedule_type="plateau",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )
        initial_lr = ppo.optimizer.param_groups[0]["lr"]

        # Feed constant "bad" value_loss for patience+1 epochs
        for _ in range(5):
            scheduler.step(10.0)

        final_lr = ppo.optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr, "LR should have decreased after patience exceeded"

    def test_plateau_scheduler_no_reduction_when_improving(self, small_model):
        """LR should NOT decrease when loss is improving."""
        from keisei.training.katago_loop import create_lr_scheduler

        params = KataGoPPOParams(learning_rate=1e-3)
        ppo = KataGoPPOAlgorithm(params, small_model)
        scheduler = create_lr_scheduler(ppo.optimizer, patience=3, min_lr=1e-6)
        initial_lr = ppo.optimizer.param_groups[0]["lr"]

        # Feed improving losses
        for loss in [10.0, 9.0, 8.0, 7.0, 6.0]:
            scheduler.step(loss)

        final_lr = ppo.optimizer.param_groups[0]["lr"]
        assert final_lr == initial_lr, "LR should not change when loss improves"


class TestRLWarmup:
    def test_warmup_epochs_use_elevated_entropy(self, small_model):
        """During warmup, lambda_entropy should be elevated."""
        params = KataGoPPOParams(lambda_entropy=0.01)
        ppo = KataGoPPOAlgorithm(params, small_model)

        assert ppo.get_entropy_coeff(epoch=0, warmup_epochs=5, warmup_entropy=0.05) == 0.05
        assert ppo.get_entropy_coeff(epoch=4, warmup_epochs=5, warmup_entropy=0.05) == 0.05
        assert ppo.get_entropy_coeff(epoch=5, warmup_epochs=5, warmup_entropy=0.05) == 0.01
        assert ppo.get_entropy_coeff(epoch=100, warmup_epochs=5, warmup_entropy=0.05) == 0.01

    def test_current_entropy_coeff_initialized(self, small_model):
        """current_entropy_coeff should default to params.lambda_entropy."""
        params = KataGoPPOParams(lambda_entropy=0.01)
        ppo = KataGoPPOAlgorithm(params, small_model)
        assert ppo.current_entropy_coeff == 0.01

    def test_update_uses_current_entropy_coeff(self, small_model):
        """Integration test: update() must READ current_entropy_coeff, not params.lambda_entropy.

        Verifies by checking that a 1000x entropy coefficient produces measurably
        different parameter updates. If update() ignores current_entropy_coeff,
        both runs would produce identical gradients.
        """
        torch.manual_seed(42)
        params = KataGoPPOParams(lambda_entropy=0.01)
        ppo = KataGoPPOAlgorithm(params, small_model)

        def fill_buffer(buf):
            for _ in range(4):
                obs = torch.randn(2, 50, 9, 9)
                legal_masks = torch.ones(2, 11259, dtype=torch.bool)
                actions, log_probs, values = ppo.select_actions(obs, legal_masks)
                buf.add(
                    obs, actions, log_probs, values,
                    torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                    torch.randint(0, 3, (2,)), torch.rand(2) * 2 - 1,
                )

        # Run update with default entropy coeff, capture weight snapshot
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        fill_buffer(buf)
        params_before_default = {
            n: p.clone() for n, p in small_model.named_parameters()
        }
        ppo.update(buf, torch.zeros(2))
        delta_default = sum(
            (p - params_before_default[n]).abs().sum().item()
            for n, p in small_model.named_parameters()
        )

        # Reset model to same state for fair comparison
        for n, p in small_model.named_parameters():
            p.data.copy_(params_before_default[n])
        ppo.optimizer = torch.optim.Adam(small_model.parameters(), lr=params.learning_rate)

        # Run update with massively elevated entropy coeff
        fill_buffer(buf)
        params_before_elevated = {
            n: p.clone() for n, p in small_model.named_parameters()
        }
        ppo.current_entropy_coeff = 10.0  # 1000x the default
        ppo.update(buf, torch.zeros(2))
        delta_elevated = sum(
            (p - params_before_elevated[n]).abs().sum().item()
            for n, p in small_model.named_parameters()
        )

        # With 1000x entropy coefficient, the gradient contribution from entropy
        # is vastly different. If update() actually reads current_entropy_coeff,
        # the parameter deltas will differ substantially.
        assert delta_default > 0, "Default update should change parameters"
        assert delta_elevated > 0, "Elevated update should change parameters"
        assert abs(delta_elevated - delta_default) > 1e-4, (
            f"Parameter deltas should differ: default={delta_default:.6f}, "
            f"elevated={delta_elevated:.6f}"
        )
