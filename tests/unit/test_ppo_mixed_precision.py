"""Tests for PPOAgent mixed precision (AMP) dtype consistency.

Reproduces the bug where experience buffer data (FP32) is mixed with
model outputs (FP16) outside the autocast scope, causing RuntimeError:
'Found dtype Float but expected Half'.
"""

import random

import numpy as np
import pytest
import torch
from torch.amp import GradScaler

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

pytestmark = pytest.mark.unit


def _make_config(tmp_path, mixed_precision=True):
    """Build a config with mixed_precision enabled."""
    return AppConfig(
        env=EnvConfig(
            device="cuda",
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
            mixed_precision=mixed_precision,
        ),
        evaluation=EvaluationConfig(num_games=2),
        logging=LoggingConfig(
            log_file=str(tmp_path / "train.log"),
            model_dir=str(tmp_path / "models"),
        ),
        wandb=WandBConfig(enabled=False),
        parallel=ParallelConfig(enabled=False),
    )


def _fill_buffer(agent, config, device):
    """Fill an experience buffer with real game data."""
    mapper = PolicyOutputMapper()
    buf = ExperienceBuffer(
        buffer_size=config.training.steps_per_epoch,
        gamma=config.training.gamma,
        lambda_gae=config.training.lambda_gae,
        device=str(device),
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

        mask = torch.zeros(mapper.get_total_actions(), dtype=torch.bool, device=device)
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
    return buf


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for mixed precision test"
)
class TestMixedPrecisionLearn:
    """PPOAgent.learn() should work with mixed precision enabled."""

    def test_learn_completes_with_mixed_precision(self, tmp_path):
        """learn() should not raise dtype mismatch under mixed precision.

        This is the core regression test for the bug:
        'Found dtype Float but expected Half'
        """
        config = _make_config(tmp_path, mixed_precision=True)
        device = torch.device("cuda")
        mapper = PolicyOutputMapper()
        model = ActorCritic(
            input_channels=46,
            num_actions_total=mapper.get_total_actions(),
        ).to(device)

        scaler = GradScaler()
        agent = PPOAgent(
            model=model,
            config=config,
            device=device,
            scaler=scaler,
            use_mixed_precision=True,
        )

        buf = _fill_buffer(agent, config, device)

        # This is where the bug manifests: learn() raises
        # RuntimeError("Found dtype Float but expected Half")
        metrics = agent.learn(buf)

        assert "ppo/policy_loss" in metrics
        assert "ppo/value_loss" in metrics
        assert isinstance(metrics["ppo/policy_loss"], float)

    def test_get_value_with_mixed_precision(self, tmp_path):
        """get_value() should work under mixed precision without dtype errors."""
        config = _make_config(tmp_path, mixed_precision=True)
        device = torch.device("cuda")
        mapper = PolicyOutputMapper()
        model = ActorCritic(
            input_channels=46,
            num_actions_total=mapper.get_total_actions(),
        ).to(device)

        scaler = GradScaler()
        agent = PPOAgent(
            model=model,
            config=config,
            device=device,
            scaler=scaler,
            use_mixed_precision=True,
        )

        obs = np.random.randn(46, 9, 9).astype(np.float32)
        value = agent.get_value(obs)
        assert isinstance(value, float)

    def test_select_action_with_mixed_precision(self, tmp_path):
        """select_action() should work under mixed precision."""
        config = _make_config(tmp_path, mixed_precision=True)
        device = torch.device("cuda")
        mapper = PolicyOutputMapper()
        model = ActorCritic(
            input_channels=46,
            num_actions_total=mapper.get_total_actions(),
        ).to(device)

        scaler = GradScaler()
        agent = PPOAgent(
            model=model,
            config=config,
            device=device,
            scaler=scaler,
            use_mixed_precision=True,
        )

        game = ShogiGame()
        obs = game.get_observation()
        legal_moves = game.get_legal_moves()
        mask = torch.zeros(mapper.get_total_actions(), dtype=torch.bool, device=device)
        for move in legal_moves:
            try:
                idx = mapper.shogi_move_to_policy_index(move)
                mask[idx] = True
            except (KeyError, ValueError):
                pass

        move, idx, log_prob, value = agent.select_action(obs, mask)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)


class TestMixedPrecisionCPUFallback:
    """Mixed precision gracefully falls back on CPU (no CUDA)."""

    def test_learn_works_on_cpu_with_mixed_precision_flag(self, tmp_path):
        """When mixed_precision=True but device=cpu, learn() should still work.

        The agent should ignore the mixed precision flag on CPU.
        """
        config = _make_config(tmp_path, mixed_precision=True)
        # Override to CPU
        config.env.device = "cpu"
        device = torch.device("cpu")
        mapper = PolicyOutputMapper()
        model = ActorCritic(
            input_channels=46,
            num_actions_total=mapper.get_total_actions(),
        )

        # Mixed precision flag on CPU should be a no-op
        agent = PPOAgent(
            model=model,
            config=config,
            device=device,
            scaler=None,
            use_mixed_precision=True,
        )

        buf = _fill_buffer(agent, config, device)
        metrics = agent.learn(buf)

        assert "ppo/policy_loss" in metrics
        assert isinstance(metrics["ppo/policy_loss"], float)
