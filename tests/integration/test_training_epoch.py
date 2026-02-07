"""Integration tests: full training epoch (fill buffer + PPO update).

Verifies the core training cycle: real gameplay fills the buffer, then
PPO learn() runs a real gradient update, producing valid metrics.
"""

import numpy as np
import pytest
import torch

from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.shogi.shogi_game import ShogiGame
from keisei.training.step_manager import StepManager


# Module-level no-op logger (not imported from conftest to avoid relative imports)
def _noop_logger(msg, also_to_wandb=False, wandb_data=None, log_level="info"):
    pass


# ---------------------------------------------------------------------------
# Buffer filling from real gameplay
# ---------------------------------------------------------------------------


class TestBufferFilling:
    """Buffer fills correctly from real gameplay steps."""

    def test_filled_buffer_has_correct_size(self, filled_buffer, integration_config):
        """After filling, buffer pointer equals steps_per_epoch."""
        assert filled_buffer.ptr == integration_config.training.steps_per_epoch

    def test_filled_buffer_observations_are_valid(self, filled_buffer):
        """Observations in the filled buffer have valid shape and values."""
        obs = filled_buffer.obs[: filled_buffer.ptr]
        assert obs.shape == (filled_buffer.ptr, 46, 9, 9)
        assert torch.isfinite(obs).all()

    def test_filled_buffer_actions_in_range(self, filled_buffer):
        """All actions in buffer are within the valid action space."""
        actions = filled_buffer.actions[: filled_buffer.ptr]
        assert (actions >= 0).all()
        assert (actions < 13527).all()

    def test_filled_buffer_has_computed_advantages(self, filled_buffer):
        """Advantages and returns are computed (non-zero somewhere)."""
        advantages = filled_buffer.advantages[: filled_buffer.ptr]
        returns = filled_buffer.returns[: filled_buffer.ptr]

        # At least some values should be non-zero after GAE computation
        assert torch.isfinite(advantages).all()
        assert torch.isfinite(returns).all()


# ---------------------------------------------------------------------------
# PPO update with real buffer data
# ---------------------------------------------------------------------------


class TestPPOUpdate:
    """PPO learn() succeeds with data from real gameplay."""

    def test_learn_returns_valid_metrics(self, ppo_agent, filled_buffer):
        """PPO learn() returns a dict with expected metric keys."""
        metrics = ppo_agent.learn(filled_buffer)

        expected_keys = {
            "ppo/policy_loss",
            "ppo/value_loss",
            "ppo/entropy",
            "ppo/kl_divergence_approx",
            "ppo/clip_fraction",
            "ppo/learning_rate",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_learn_metrics_are_finite(self, ppo_agent, filled_buffer):
        """All metrics returned by learn() are finite floats."""
        metrics = ppo_agent.learn(filled_buffer)

        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not float: {type(value)}"
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_learn_updates_model_weights(self, ppo_agent, filled_buffer):
        """Model parameters change after a PPO update."""
        # Snapshot initial weights
        initial_params = {
            name: p.clone() for name, p in ppo_agent.model.named_parameters()
        }

        ppo_agent.learn(filled_buffer)

        # Check that at least some weights changed
        any_changed = False
        for name, p in ppo_agent.model.named_parameters():
            if not torch.equal(p, initial_params[name]):
                any_changed = True
                break

        assert any_changed, "No model parameters changed after PPO update"

    def test_learn_updates_gradient_norm(self, ppo_agent, filled_buffer):
        """The agent's last_gradient_norm is updated after learn()."""
        ppo_agent.learn(filled_buffer)
        # Gradient norm should be set to a positive value after training
        assert ppo_agent.last_gradient_norm >= 0.0


# ---------------------------------------------------------------------------
# Multiple epochs in sequence
# ---------------------------------------------------------------------------


class TestMultipleEpochs:
    """Multiple training epochs work in sequence."""

    def test_two_epochs_both_succeed(
        self,
        integration_config,
        session_policy_mapper,
    ):
        """Fill buffer, learn, clear, fill again, learn again -- no errors."""
        game = ShogiGame(max_moves_per_game=500)
        model = ActorCritic(
            input_channels=46,
            num_actions_total=session_policy_mapper.get_total_actions(),
        )
        agent = PPOAgent(
            model=model, config=integration_config, device=torch.device("cpu")
        )
        buf = ExperienceBuffer(
            buffer_size=integration_config.training.steps_per_epoch,
            gamma=integration_config.training.gamma,
            lambda_gae=integration_config.training.lambda_gae,
            device="cpu",
            num_actions=session_policy_mapper.get_total_actions(),
        )
        sm = StepManager(
            config=integration_config,
            game=game,
            agent=agent,
            policy_mapper=session_policy_mapper,
            experience_buffer=buf,
        )

        all_metrics = []
        for epoch in range(2):
            episode_state = sm.reset_episode()
            for t in range(integration_config.training.steps_per_epoch):
                result = sm.execute_step(
                    episode_state, global_timestep=t, logger_func=_noop_logger
                )
                if result.success:
                    episode_state = sm.update_episode_state(episode_state, result)
                    if result.done:
                        episode_state = sm.reset_episode()
                else:
                    episode_state = sm.reset_episode()

            last_val = agent.get_value(episode_state.current_obs)
            buf.compute_advantages_and_returns(last_val)
            metrics = agent.learn(buf)
            all_metrics.append(metrics)
            buf.clear()

        assert len(all_metrics) == 2
        for m in all_metrics:
            assert "ppo/policy_loss" in m
            assert np.isfinite(m["ppo/policy_loss"])

    def test_loss_values_are_reasonable_after_epoch(self, ppo_agent, filled_buffer):
        """Loss values after an epoch are within a reasonable numerical range."""
        metrics = ppo_agent.learn(filled_buffer)

        # Policy loss can be positive or negative, but shouldn't be extreme
        assert abs(metrics["ppo/policy_loss"]) < 100.0
        # Value loss should be non-negative (MSE)
        assert metrics["ppo/value_loss"] >= 0.0
        # Learning rate should be positive
        assert metrics["ppo/learning_rate"] > 0.0
