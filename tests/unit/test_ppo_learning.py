"""Tests for PPOAgent.learn() with a filled experience buffer."""

import copy
import random

import numpy as np
import pytest
import torch

from keisei.config_schema import (
    AppConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    ParallelConfig,
    TrainingConfig,
    WandBConfig,
)
from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils.utils import PolicyOutputMapper


def make_config(tmp_path):
    """Build a minimal AppConfig pointing at tmp_path for logs/models."""
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            input_channels=46,
            num_actions_total=13527,
            seed=42,
        ),
        training=TrainingConfig(
            total_timesteps=200,
            steps_per_epoch=16,
            ppo_epochs=2,
            minibatch_size=8,
            learning_rate=3e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=100,
            evaluation_interval_timesteps=100,
            enable_torch_compile=False,
        ),
        evaluation=EvaluationConfig(num_games=2),
        logging=LoggingConfig(
            log_file=str(tmp_path / "train.log"),
            model_dir=str(tmp_path / "models"),
        ),
        wandb=WandBConfig(enabled=False),
        parallel=ParallelConfig(enabled=False),
    )


@pytest.fixture
def filled_buffer(tmp_path):
    """
    Create a PPOAgent and an ExperienceBuffer filled with real game data.
    Returns (buffer, agent) so tests can exercise learn().
    """
    mapper = PolicyOutputMapper()
    config = make_config(tmp_path)
    model = ActorCritic(
        input_channels=46,
        num_actions_total=mapper.get_total_actions(),
    )
    agent = PPOAgent(
        model=model, config=config, device=torch.device("cpu")
    )
    buf = ExperienceBuffer(
        buffer_size=config.training.steps_per_epoch,
        gamma=config.training.gamma,
        lambda_gae=config.training.lambda_gae,
        device="cpu",
        num_actions=mapper.get_total_actions(),
    )
    game = ShogiGame()
    rng = random.Random(42)

    for _ in range(config.training.steps_per_epoch):
        obs = game.get_observation()
        legal_moves = game.get_legal_moves()
        if not legal_moves or game.game_over:
            game.reset()
            legal_moves = game.get_legal_moves()
            obs = game.get_observation()

        mask = torch.zeros(mapper.get_total_actions(), dtype=torch.bool)
        for move in legal_moves:
            try:
                idx = mapper.shogi_move_to_policy_index(move)
                mask[idx] = True
            except (KeyError, ValueError):
                pass

        shogi_move, action_idx, log_prob, value = agent.select_action(
            obs, mask
        )
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

        if shogi_move and shogi_move in legal_moves:
            _, reward, done, _ = game.make_move(shogi_move)
        else:
            fallback = rng.choice(legal_moves)
            _, reward, done, _ = game.make_move(fallback)

        buf.add(obs_tensor, action_idx, reward, log_prob, value, done, mask)

    buf.compute_advantages_and_returns(last_value=0.0)
    return buf, agent


# ---------------------------------------------------------------------------
# 1. Learn returns metrics
# ---------------------------------------------------------------------------


class TestLearnReturnsMetrics:
    """Verify learn() returns properly structured metric dicts."""

    EXPECTED_KEYS = {
        "ppo/policy_loss",
        "ppo/value_loss",
        "ppo/entropy",
        "ppo/kl_divergence_approx",
        "ppo/clip_fraction",
        "ppo/learning_rate",
    }

    def test_learn_returns_all_keys(self, filled_buffer):
        """learn() returns a dict containing every expected metric key."""
        buf, agent = filled_buffer
        metrics = agent.learn(buf)
        assert isinstance(metrics, dict)
        assert self.EXPECTED_KEYS.issubset(metrics.keys())

    def test_all_metrics_finite(self, filled_buffer):
        """Every returned metric value is finite (not nan or inf)."""
        buf, agent = filled_buffer
        metrics = agent.learn(buf)
        for key, value in metrics.items():
            assert np.isfinite(value), (
                f"Metric {key} is not finite: {value}"
            )

    def test_policy_loss_is_float(self, filled_buffer):
        """policy_loss is a plain float."""
        buf, agent = filled_buffer
        metrics = agent.learn(buf)
        assert isinstance(metrics["ppo/policy_loss"], float)

    def test_value_loss_non_negative(self, filled_buffer):
        """value_loss (MSE-based) is non-negative."""
        buf, agent = filled_buffer
        metrics = agent.learn(buf)
        assert metrics["ppo/value_loss"] >= 0.0

    def test_learning_rate_matches_config(self, filled_buffer):
        """Reported learning_rate matches the configured initial LR (no scheduler)."""
        buf, agent = filled_buffer
        metrics = agent.learn(buf)
        # Without a scheduler, LR should remain at the configured value
        assert metrics["ppo/learning_rate"] == pytest.approx(3e-4, rel=1e-4)


# ---------------------------------------------------------------------------
# 2. Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify that learn() actually updates the model."""

    def test_parameters_change_after_learn(self, filled_buffer):
        """Model parameters change after a learn() call."""
        buf, agent = filled_buffer
        # Snapshot parameters before learning
        params_before = {
            name: p.detach().clone()
            for name, p in agent.model.named_parameters()
        }
        agent.learn(buf)
        params_after = {
            name: p.detach().clone()
            for name, p in agent.model.named_parameters()
        }
        any_changed = any(
            not torch.equal(params_before[name], params_after[name])
            for name in params_before
        )
        assert any_changed, "No model parameters changed after learn()"

    def test_gradient_norm_tracked(self, filled_buffer):
        """After learn(), agent.last_gradient_norm > 0."""
        buf, agent = filled_buffer
        agent.learn(buf)
        assert agent.last_gradient_norm > 0.0

    def test_kl_divergence_tracked(self, filled_buffer):
        """Metrics include ppo/kl_divergence_approx."""
        buf, agent = filled_buffer
        metrics = agent.learn(buf)
        assert "ppo/kl_divergence_approx" in metrics
        assert isinstance(metrics["ppo/kl_divergence_approx"], float)

    def test_clip_fraction_in_range(self, filled_buffer):
        """Clip fraction is between 0 and 1 (inclusive)."""
        buf, agent = filled_buffer
        metrics = agent.learn(buf)
        clip_frac = metrics["ppo/clip_fraction"]
        assert 0.0 <= clip_frac <= 1.0, (
            f"clip_fraction out of range: {clip_frac}"
        )


# ---------------------------------------------------------------------------
# 3. Multiple learn calls
# ---------------------------------------------------------------------------


class TestMultipleLearnCalls:
    """Verify behaviour across repeated learn() invocations."""

    def test_multiple_learn_calls_succeed(self, filled_buffer):
        """Can call learn() multiple times on the same buffer without error."""
        buf, agent = filled_buffer
        metrics_1 = agent.learn(buf)
        metrics_2 = agent.learn(buf)
        assert isinstance(metrics_1, dict)
        assert isinstance(metrics_2, dict)

    def test_loss_changes_across_learn_calls(self, filled_buffer):
        """Losses change (decrease or shift) after successive learn() calls."""
        buf, agent = filled_buffer
        metrics_1 = agent.learn(buf)
        metrics_2 = agent.learn(buf)
        # At least one of policy_loss or value_loss should differ
        policy_changed = (
            metrics_1["ppo/policy_loss"] != metrics_2["ppo/policy_loss"]
        )
        value_changed = (
            metrics_1["ppo/value_loss"] != metrics_2["ppo/value_loss"]
        )
        assert policy_changed or value_changed, (
            "Neither policy_loss nor value_loss changed between learn calls"
        )

    def test_learning_rate_with_scheduler(self, tmp_path):
        """With a linear scheduler, LR changes after learn()."""
        mapper = PolicyOutputMapper()
        config = make_config(tmp_path)
        # Enable a linear LR schedule stepping on each epoch
        config_dict = config.model_dump()
        config_dict["training"]["lr_schedule_type"] = "linear"
        config_dict["training"]["lr_schedule_step_on"] = "epoch"
        config_dict["training"]["lr_schedule_kwargs"] = {
            "final_lr_fraction": 0.1,
        }
        config = AppConfig(**config_dict)

        model = ActorCritic(
            input_channels=46,
            num_actions_total=mapper.get_total_actions(),
        )
        agent = PPOAgent(
            model=model, config=config, device=torch.device("cpu")
        )

        # Fill a buffer
        buf = ExperienceBuffer(
            buffer_size=config.training.steps_per_epoch,
            gamma=config.training.gamma,
            lambda_gae=config.training.lambda_gae,
            device="cpu",
            num_actions=mapper.get_total_actions(),
        )
        game = ShogiGame()
        rng = random.Random(99)
        for _ in range(config.training.steps_per_epoch):
            obs = game.get_observation()
            legal_moves = game.get_legal_moves()
            if not legal_moves or game.game_over:
                game.reset()
                legal_moves = game.get_legal_moves()
                obs = game.get_observation()
            mask = torch.zeros(mapper.get_total_actions(), dtype=torch.bool)
            for move in legal_moves:
                try:
                    idx = mapper.shogi_move_to_policy_index(move)
                    mask[idx] = True
                except (KeyError, ValueError):
                    pass
            shogi_move, action_idx, log_prob, value = agent.select_action(
                obs, mask
            )
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            if shogi_move and shogi_move in legal_moves:
                _, reward, done, _ = game.make_move(shogi_move)
            else:
                fallback = rng.choice(legal_moves)
                _, reward, done, _ = game.make_move(fallback)
            buf.add(
                obs_tensor, action_idx, reward, log_prob, value, done, mask
            )
        buf.compute_advantages_and_returns(last_value=0.0)

        lr_before = agent.optimizer.param_groups[0]["lr"]
        agent.learn(buf)
        lr_after = agent.optimizer.param_groups[0]["lr"]

        assert lr_after < lr_before, (
            f"LR should decrease with linear schedule: {lr_before} -> {lr_after}"
        )


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case and boundary scenarios for learn()."""

    def test_empty_buffer_returns_defaults(self, tmp_path):
        """learn() with an empty buffer returns a dict with zeros."""
        mapper = PolicyOutputMapper()
        config = make_config(tmp_path)
        model = ActorCritic(
            input_channels=46,
            num_actions_total=mapper.get_total_actions(),
        )
        agent = PPOAgent(
            model=model, config=config, device=torch.device("cpu")
        )
        empty_buf = ExperienceBuffer(
            buffer_size=config.training.steps_per_epoch,
            gamma=config.training.gamma,
            lambda_gae=config.training.lambda_gae,
            device="cpu",
            num_actions=mapper.get_total_actions(),
        )
        # get_batch on empty buffer returns {}; learn handles that gracefully
        metrics = agent.learn(empty_buf)
        assert isinstance(metrics, dict)
        assert metrics["ppo/policy_loss"] == 0.0
        assert metrics["ppo/value_loss"] == 0.0
        assert metrics["ppo/entropy"] == 0.0

    def test_advantage_normalization_no_nan(self, filled_buffer):
        """Advantage normalization does not produce NaN in metrics."""
        buf, agent = filled_buffer
        metrics = agent.learn(buf)
        for key, value in metrics.items():
            assert not np.isnan(value), (
                f"NaN found in metric {key} after advantage normalization"
            )

    def test_entropy_is_non_negative(self, filled_buffer):
        """Entropy loss term (negated entropy) can be any sign,
        but raw entropy should be non-negative. The metric
        ppo/entropy stores the negated mean entropy used in the loss
        so we check it is finite; the underlying distribution entropy >= 0."""
        buf, agent = filled_buffer
        metrics = agent.learn(buf)
        # ppo/entropy is -mean(entropy) in the code, so it should be <= 0
        # (or close to 0). It must at least be finite.
        assert np.isfinite(metrics["ppo/entropy"]), (
            f"Entropy metric is not finite: {metrics['ppo/entropy']}"
        )
        # The underlying distribution entropy is non-negative,
        # so the negated value should be non-positive.
        assert metrics["ppo/entropy"] <= 1e-6, (
            f"Negated entropy should be <= 0, got {metrics['ppo/entropy']}"
        )
