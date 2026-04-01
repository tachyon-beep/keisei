"""Tests for LR scheduling in KataGoTrainingLoop."""

import pytest
import torch

from keisei.training.katago_ppo import KataGoPPOAlgorithm, KataGoPPOParams
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
