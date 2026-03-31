import pytest
import torch
import numpy as np
from keisei.training.ppo import compute_gae, RolloutBuffer, PPOAlgorithm
from keisei.training.algorithm_registry import PPOParams


class TestGAE:
    def test_single_step_terminal(self) -> None:
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5])
        dones = torch.tensor([True])
        next_value = torch.tensor(0.0)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert abs(advantages[0].item() - 0.5) < 1e-5

    def test_two_steps_no_terminal(self) -> None:
        rewards = torch.tensor([0.0, 0.0])
        values = torch.tensor([0.5, 0.6])
        dones = torch.tensor([False, False])
        next_value = torch.tensor(0.7)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert advantages.shape == (2,)
        assert abs(advantages[1].item() - 0.093) < 1e-3
        assert abs(advantages[0].item() - 0.1815) < 1e-2

    def test_terminal_resets_bootstrap(self) -> None:
        rewards = torch.tensor([1.0, 0.0])
        values = torch.tensor([0.3, 0.4])
        dones = torch.tensor([True, False])
        next_value = torch.tensor(0.5)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert abs(advantages[0].item() - 0.7) < 0.1


class TestRolloutBuffer:
    def test_add_and_get(self) -> None:
        buf = RolloutBuffer(num_envs=2, obs_shape=(46, 9, 9), action_space=13527)
        obs = torch.randn(2, 46, 9, 9)
        actions = torch.tensor([0, 1])
        log_probs = torch.tensor([-1.0, -2.0])
        values = torch.tensor([0.5, 0.6])
        rewards = torch.tensor([0.0, 0.0])
        dones = torch.tensor([False, False])
        legal_masks = torch.ones(2, 13527, dtype=torch.bool)

        buf.add(obs, actions, log_probs, values, rewards, dones, legal_masks)
        assert buf.size == 1
        buf.add(obs, actions, log_probs, values, rewards, dones, legal_masks)
        assert buf.size == 2

    def test_clear(self) -> None:
        buf = RolloutBuffer(num_envs=2, obs_shape=(46, 9, 9), action_space=13527)
        obs = torch.randn(2, 46, 9, 9)
        buf.add(obs, torch.zeros(2, dtype=torch.long), torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2, dtype=torch.bool), torch.ones(2, 13527, dtype=torch.bool))
        buf.clear()
        assert buf.size == 0


class TestPPOAlgorithm:
    def test_select_actions(self) -> None:
        from keisei.training.models.resnet import ResNetModel, ResNetParams
        params = PPOParams(learning_rate=1e-3, batch_size=4, epochs_per_batch=1)
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
        ppo = PPOAlgorithm(params, model)

        obs = torch.randn(4, 46, 9, 9)
        legal_masks = torch.ones(4, 13527, dtype=torch.bool)
        actions, log_probs, values = ppo.select_actions(obs, legal_masks)
        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)
        assert (actions >= 0).all()
        assert (actions < 13527).all()

    def test_update_returns_losses(self) -> None:
        from keisei.training.models.resnet import ResNetModel, ResNetParams
        params = PPOParams(learning_rate=1e-3, batch_size=4, epochs_per_batch=1)
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
        ppo = PPOAlgorithm(params, model)

        buf = RolloutBuffer(num_envs=2, obs_shape=(46, 9, 9), action_space=13527)
        for _ in range(8):
            obs = torch.randn(2, 46, 9, 9)
            legal_masks = torch.ones(2, 13527, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(obs, actions, log_probs, values, torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks)

        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "entropy" in losses
        assert "gradient_norm" in losses
        assert isinstance(losses["policy_loss"], float)
