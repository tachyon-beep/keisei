"""Tests for PPOAgent save/load checkpoint roundtrip."""

import os

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
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.utils.utils import PolicyOutputMapper


def make_config(tmp_path, lr_schedule_type=None, lr_schedule_step_on="epoch"):
    """Build a minimal AppConfig, optionally with LR scheduling."""
    training_kwargs = dict(
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
        lr_schedule_type=lr_schedule_type,
        lr_schedule_step_on=lr_schedule_step_on,
    )
    if lr_schedule_type == "linear":
        training_kwargs["lr_schedule_kwargs"] = {
            "final_lr_fraction": 0.1,
        }
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            input_channels=46,
            num_actions_total=13527,
            seed=42,
        ),
        training=TrainingConfig(**training_kwargs),
        evaluation=EvaluationConfig(num_games=2),
        logging=LoggingConfig(
            log_file=str(tmp_path / "train.log"),
            model_dir=str(tmp_path / "models"),
        ),
        wandb=WandBConfig(enabled=False),
        parallel=ParallelConfig(enabled=False),
    )


@pytest.fixture
def mapper():
    return PolicyOutputMapper()


@pytest.fixture
def agent_and_path(tmp_path, mapper):
    """Create an agent and return (agent, checkpoint_path)."""
    config = make_config(tmp_path)
    model = ActorCritic(
        input_channels=46,
        num_actions_total=mapper.get_total_actions(),
    )
    agent = PPOAgent(
        model=model, config=config, device=torch.device("cpu")
    )
    ckpt_path = str(tmp_path / "checkpoint.pt")
    return agent, ckpt_path


# ---------------------------------------------------------------------------
# 1. Save/load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    """Verify model weights and metadata survive a save/load cycle."""

    def test_weights_restored_same_predictions(
        self, agent_and_path, mapper, tmp_path
    ):
        """After save then load, the model gives the same predictions."""
        agent, ckpt_path = agent_and_path
        # Generate a deterministic input
        obs = torch.randn(1, 46, 9, 9)
        with torch.no_grad():
            logits_before, value_before = agent.model(obs)

        agent.save_model(ckpt_path, global_timestep=100)

        # Create a fresh agent and load
        config = make_config(tmp_path)
        model2 = ActorCritic(
            input_channels=46,
            num_actions_total=mapper.get_total_actions(),
        )
        agent2 = PPOAgent(
            model=model2, config=config, device=torch.device("cpu")
        )
        agent2.load_model(ckpt_path)

        with torch.no_grad():
            logits_after, value_after = agent2.model(obs)

        assert torch.allclose(logits_before, logits_after, atol=1e-6)
        assert torch.allclose(value_before, value_after, atol=1e-6)

    def test_global_timestep_preserved(self, agent_and_path):
        """Loaded checkpoint contains the correct global_timestep."""
        agent, ckpt_path = agent_and_path
        agent.save_model(ckpt_path, global_timestep=42, total_episodes_completed=7)
        result = agent.load_model(ckpt_path)
        assert result["global_timestep"] == 42
        assert result["total_episodes_completed"] == 7

    def test_optimizer_state_preserved(
        self, agent_and_path, mapper, tmp_path
    ):
        """Optimizer learning rate is preserved through save/load."""
        agent, ckpt_path = agent_and_path

        # Get current LR
        lr_before = agent.optimizer.param_groups[0]["lr"]
        agent.save_model(ckpt_path)

        # Load into a fresh agent
        config = make_config(tmp_path)
        model2 = ActorCritic(
            input_channels=46,
            num_actions_total=mapper.get_total_actions(),
        )
        agent2 = PPOAgent(
            model=model2, config=config, device=torch.device("cpu")
        )
        agent2.load_model(ckpt_path)
        lr_after = agent2.optimizer.param_groups[0]["lr"]

        assert lr_before == pytest.approx(lr_after, rel=1e-6)

    def test_stats_to_save_preserved(self, agent_and_path):
        """Extra stats passed via stats_to_save appear in checkpoint."""
        agent, ckpt_path = agent_and_path
        stats = {"black_wins": 10, "white_wins": 5, "draws": 3}
        agent.save_model(
            ckpt_path,
            global_timestep=50,
            total_episodes_completed=18,
            stats_to_save=stats,
        )
        result = agent.load_model(ckpt_path)
        assert result["black_wins"] == 10
        assert result["white_wins"] == 5
        assert result["draws"] == 3


# ---------------------------------------------------------------------------
# 2. File handling
# ---------------------------------------------------------------------------


class TestFileHandling:
    """Verify file system interactions for save/load."""

    def test_load_nonexistent_returns_error(self, agent_and_path):
        """Loading a non-existent file returns an error dict (no exception)."""
        agent, _ = agent_and_path
        result = agent.load_model("/tmp/nonexistent_checkpoint_xyz.pt")
        assert isinstance(result, dict)
        assert "error" in result
        assert result["global_timestep"] == 0

    def test_save_creates_file(self, agent_and_path):
        """save_model creates a file at the specified path."""
        agent, ckpt_path = agent_and_path
        assert not os.path.exists(ckpt_path)
        agent.save_model(ckpt_path)
        assert os.path.exists(ckpt_path)
        assert os.path.getsize(ckpt_path) > 0

    def test_saved_file_is_valid_torch_checkpoint(self, agent_and_path):
        """The saved file can be loaded by torch.load."""
        agent, ckpt_path = agent_and_path
        agent.save_model(ckpt_path, global_timestep=99)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert isinstance(checkpoint, dict)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["global_timestep"] == 99


# ---------------------------------------------------------------------------
# 3. Scheduler state
# ---------------------------------------------------------------------------


class TestSchedulerState:
    """Verify LR scheduler state is correctly saved and restored."""

    def test_scheduler_state_saved_and_loaded(self, tmp_path, mapper):
        """Scheduler state dict is saved and successfully loaded."""
        config = make_config(
            tmp_path,
            lr_schedule_type="linear",
            lr_schedule_step_on="epoch",
        )
        model = ActorCritic(
            input_channels=46,
            num_actions_total=mapper.get_total_actions(),
        )
        agent = PPOAgent(
            model=model, config=config, device=torch.device("cpu")
        )
        assert agent.scheduler is not None

        # Step the scheduler a few times to change its internal state
        for _ in range(3):
            agent.scheduler.step()
        lr_after_steps = agent.optimizer.param_groups[0]["lr"]

        ckpt_path = str(tmp_path / "sched_ckpt.pt")
        agent.save_model(ckpt_path, global_timestep=30)

        # Load into fresh agent with same scheduler config
        config2 = make_config(
            tmp_path,
            lr_schedule_type="linear",
            lr_schedule_step_on="epoch",
        )
        model2 = ActorCritic(
            input_channels=46,
            num_actions_total=mapper.get_total_actions(),
        )
        agent2 = PPOAgent(
            model=model2, config=config2, device=torch.device("cpu")
        )
        agent2.load_model(ckpt_path)

        lr_loaded = agent2.optimizer.param_groups[0]["lr"]
        assert lr_loaded == pytest.approx(lr_after_steps, rel=1e-6)

    def test_scheduler_continues_from_saved_state(self, tmp_path, mapper):
        """After loading, further scheduler steps produce the same LR trajectory."""
        config = make_config(
            tmp_path,
            lr_schedule_type="linear",
            lr_schedule_step_on="epoch",
        )
        model = ActorCritic(
            input_channels=46,
            num_actions_total=mapper.get_total_actions(),
        )
        agent = PPOAgent(
            model=model, config=config, device=torch.device("cpu")
        )
        # Step scheduler twice
        agent.scheduler.step()
        agent.scheduler.step()

        ckpt_path = str(tmp_path / "sched_continue.pt")
        agent.save_model(ckpt_path)

        # Continue stepping original agent
        agent.scheduler.step()
        expected_lr = agent.optimizer.param_groups[0]["lr"]

        # Load and step the loaded agent once
        config2 = make_config(
            tmp_path,
            lr_schedule_type="linear",
            lr_schedule_step_on="epoch",
        )
        model2 = ActorCritic(
            input_channels=46,
            num_actions_total=mapper.get_total_actions(),
        )
        agent2 = PPOAgent(
            model=model2, config=config2, device=torch.device("cpu")
        )
        agent2.load_model(ckpt_path)
        agent2.scheduler.step()
        loaded_lr = agent2.optimizer.param_groups[0]["lr"]

        assert loaded_lr == pytest.approx(expected_lr, rel=1e-6)

    def test_lr_schedule_type_preserved(self, tmp_path, mapper):
        """lr_schedule_type is stored in the checkpoint and returned on load."""
        config = make_config(
            tmp_path,
            lr_schedule_type="cosine",
            lr_schedule_step_on="update",
        )
        model = ActorCritic(
            input_channels=46,
            num_actions_total=mapper.get_total_actions(),
        )
        agent = PPOAgent(
            model=model, config=config, device=torch.device("cpu")
        )
        ckpt_path = str(tmp_path / "type_ckpt.pt")
        agent.save_model(ckpt_path)

        result = agent.load_model(ckpt_path)
        assert result["lr_schedule_type"] == "cosine"
        assert result["lr_schedule_step_on"] == "update"
