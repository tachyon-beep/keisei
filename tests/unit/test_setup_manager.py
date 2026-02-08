"""Unit tests for keisei.training.setup_manager: SetupManager."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch

from keisei.config_schema import (
    AppConfig,
    DisplayConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    ParallelConfig,
    TrainingConfig,
    WandBConfig,
    WebUIConfig,
)
from keisei.training.setup_manager import SetupManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(*, tmp_path=None):
    """Build a minimal AppConfig suitable for SetupManager unit tests."""
    log_dir = str(tmp_path) if tmp_path else "/tmp/test_setup_manager"
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            input_channels=46,
            num_actions_total=13527,
            seed=42,
        ),
        training=TrainingConfig(
            total_timesteps=100,
            steps_per_epoch=16,
            ppo_epochs=2,
            minibatch_size=8,
            learning_rate=3e-4,
            model_type="resnet",
            input_features="core46",
            tower_depth=1,
            tower_width=16,
            se_ratio=0.0,
            gamma=0.99,
            lambda_gae=0.95,
        ),
        evaluation=EvaluationConfig(num_games=2, max_moves_per_game=50),
        logging=LoggingConfig(
            log_file=os.path.join(log_dir, "train.log"),
            model_dir=os.path.join(log_dir, "models"),
        ),
        wandb=WandBConfig(enabled=False),
        parallel=ParallelConfig(enabled=False),
        display=DisplayConfig(),
        webui=WebUIConfig(enabled=False),
    )


def _noop_logger(*a, **kw):
    """Silent logger for test usage."""
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    return _make_config(tmp_path=tmp_path)


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


@pytest.fixture
def manager(config, cpu_device):
    return SetupManager(config, cpu_device)


# ===========================================================================
# TestInit
# ===========================================================================


class TestInit:
    """Verify SetupManager stores config and device."""

    def test_stores_config(self, manager, config):
        assert manager.config is config

    def test_stores_device(self, manager, cpu_device):
        assert manager.device == cpu_device


# ===========================================================================
# TestSetupGameComponents
# ===========================================================================


class TestSetupGameComponents:
    """Tests for setup_game_components."""

    def test_returns_game_and_mapper(self, manager):
        """Returns tuple of (game, mapper, action_space_size, obs_space_shape)."""
        env_manager = MagicMock()
        mock_game = MagicMock()
        mock_mapper = MagicMock()
        env_manager.setup_environment.return_value = (mock_game, mock_mapper)
        env_manager.action_space_size = 13527
        env_manager.obs_space_shape = (46, 9, 9)

        game, mapper, action_size, obs_shape = manager.setup_game_components(
            env_manager
        )
        assert game is mock_game
        assert mapper is mock_mapper
        assert action_size == 13527
        assert obs_shape == (46, 9, 9)

    def test_raises_when_game_is_none(self, manager):
        """Raises RuntimeError when env_manager returns None game."""
        env_manager = MagicMock()
        env_manager.setup_environment.return_value = (None, MagicMock())
        env_manager.action_space_size = 0
        env_manager.obs_space_shape = ()

        with pytest.raises(RuntimeError, match="Failed to initialize game components"):
            manager.setup_game_components(env_manager)

    def test_raises_when_mapper_is_none(self, manager):
        """Raises RuntimeError when env_manager returns None mapper."""
        env_manager = MagicMock()
        env_manager.setup_environment.return_value = (MagicMock(), None)
        env_manager.action_space_size = 0
        env_manager.obs_space_shape = ()

        with pytest.raises(RuntimeError, match="Failed to initialize game components"):
            manager.setup_game_components(env_manager)

    def test_wraps_env_manager_error(self, manager):
        """Wraps OSError from env_manager in RuntimeError."""
        env_manager = MagicMock()
        env_manager.setup_environment.side_effect = OSError("env failed")

        with pytest.raises(RuntimeError, match="Failed to initialize game components"):
            manager.setup_game_components(env_manager)


# ===========================================================================
# TestSetupTrainingComponents
# ===========================================================================


class TestSetupTrainingComponents:
    """Tests for setup_training_components."""

    @patch("keisei.training.setup_manager.PPOAgent")
    @patch("keisei.training.setup_manager.ExperienceBuffer")
    def test_returns_model_agent_buffer(self, mock_buffer_cls, mock_agent_cls, manager):
        """Returns tuple of (model, agent, experience_buffer)."""
        model_manager = MagicMock()
        mock_model = MagicMock()
        model_manager.create_model.return_value = mock_model
        model_manager.scaler = None
        model_manager.use_mixed_precision = False

        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_buffer = MagicMock()
        mock_buffer_cls.return_value = mock_buffer

        model, agent, buffer = manager.setup_training_components(model_manager)
        assert model is mock_model
        assert agent is mock_agent
        assert buffer is mock_buffer

    @patch("keisei.training.setup_manager.PPOAgent")
    @patch("keisei.training.setup_manager.ExperienceBuffer")
    def test_raises_when_model_is_none(
        self, mock_buffer_cls, mock_agent_cls, manager
    ):
        """Raises RuntimeError when model_manager.create_model returns None."""
        model_manager = MagicMock()
        model_manager.create_model.return_value = None

        with pytest.raises(RuntimeError, match="Model was not created"):
            manager.setup_training_components(model_manager)


# ===========================================================================
# TestSetupStepManager
# ===========================================================================


class TestSetupStepManager:
    """Tests for setup_step_manager."""

    @patch("keisei.training.setup_manager.StepManager")
    def test_creates_step_manager_with_correct_args(self, mock_sm_cls, manager):
        """StepManager is instantiated with config, game, agent, mapper, buffer."""
        mock_game = MagicMock()
        mock_agent = MagicMock()
        mock_mapper = MagicMock()
        mock_buffer = MagicMock()
        mock_sm = MagicMock()
        mock_sm_cls.return_value = mock_sm

        result = manager.setup_step_manager(
            mock_game, mock_agent, mock_mapper, mock_buffer
        )
        assert result is mock_sm
        mock_sm_cls.assert_called_once_with(
            config=manager.config,
            game=mock_game,
            agent=mock_agent,
            policy_mapper=mock_mapper,
            experience_buffer=mock_buffer,
        )


# ===========================================================================
# TestHandleCheckpointResume
# ===========================================================================


class TestHandleCheckpointResume:
    """Tests for handle_checkpoint_resume."""

    def test_raises_when_agent_is_none(self, manager):
        """Raises RuntimeError when agent is None/falsy."""
        model_manager = MagicMock()
        metrics_manager = MagicMock()
        mock_logger = MagicMock()

        with pytest.raises(RuntimeError, match="Agent not initialized"):
            manager.handle_checkpoint_resume(
                model_manager=model_manager,
                agent=None,
                model_dir="/tmp/models",
                resume_path_override=None,
                metrics_manager=metrics_manager,
                logger=mock_logger,
            )

    def test_delegates_to_model_manager(self, manager):
        """Calls model_manager.handle_checkpoint_resume with correct args."""
        model_manager = MagicMock()
        model_manager.resumed_from_checkpoint = False
        model_manager.checkpoint_data = None
        agent = MagicMock()
        metrics_manager = MagicMock()
        mock_logger = MagicMock()

        manager.handle_checkpoint_resume(
            model_manager=model_manager,
            agent=agent,
            model_dir="/tmp/models",
            resume_path_override="/path/to/ckpt.pth",
            metrics_manager=metrics_manager,
            logger=mock_logger,
        )
        model_manager.handle_checkpoint_resume.assert_called_once_with(
            agent=agent,
            model_dir="/tmp/models",
            resume_path_override="/path/to/ckpt.pth",
        )

    def test_restores_metrics_from_checkpoint_data(self, manager):
        """Calls metrics_manager.restore_from_checkpoint when data exists."""
        model_manager = MagicMock()
        checkpoint_data = {"global_timestep": 500, "episodes": 10}
        model_manager.checkpoint_data = checkpoint_data
        model_manager.resumed_from_checkpoint = True
        agent = MagicMock()
        metrics_manager = MagicMock()
        mock_logger = MagicMock()

        manager.handle_checkpoint_resume(
            model_manager=model_manager,
            agent=agent,
            model_dir="/tmp/models",
            resume_path_override=None,
            metrics_manager=metrics_manager,
            logger=mock_logger,
        )
        metrics_manager.restore_from_checkpoint.assert_called_once_with(checkpoint_data)

    def test_no_restore_when_no_checkpoint_data(self, manager):
        """Does not call restore when checkpoint_data is None."""
        model_manager = MagicMock()
        model_manager.checkpoint_data = None
        model_manager.resumed_from_checkpoint = False
        agent = MagicMock()
        metrics_manager = MagicMock()
        mock_logger = MagicMock()

        manager.handle_checkpoint_resume(
            model_manager=model_manager,
            agent=agent,
            model_dir="/tmp/models",
            resume_path_override=None,
            metrics_manager=metrics_manager,
            logger=mock_logger,
        )
        metrics_manager.restore_from_checkpoint.assert_not_called()

    def test_returns_resumed_from_checkpoint_value(self, manager):
        """Returns the value of model_manager.resumed_from_checkpoint."""
        model_manager = MagicMock()
        model_manager.resumed_from_checkpoint = "/path/to/ckpt.pth"
        model_manager.checkpoint_data = None
        agent = MagicMock()
        metrics_manager = MagicMock()
        mock_logger = MagicMock()

        result = manager.handle_checkpoint_resume(
            model_manager=model_manager,
            agent=agent,
            model_dir="/tmp/models",
            resume_path_override=None,
            metrics_manager=metrics_manager,
            logger=mock_logger,
        )
        assert result == "/path/to/ckpt.pth"


# ===========================================================================
# TestLogEvent
# ===========================================================================


class TestLogEvent:
    """Tests for log_event."""

    def test_writes_to_file(self, manager, tmp_path):
        """Writes timestamped message to the log file."""
        log_file = str(tmp_path / "test.log")
        manager.log_event("Test message", log_file)

        with open(log_file) as f:
            content = f.read()
        assert "Test message" in content

    def test_handles_io_error_gracefully(self, manager):
        """Does not raise on IOError (invalid path)."""
        # Should not raise
        manager.log_event("Test message", "/nonexistent/dir/log.txt")


# ===========================================================================
# TestLogRunInfo
# ===========================================================================


class TestLogRunInfo:
    """Tests for log_run_info."""

    def test_delegates_to_session_and_model_managers(self, manager):
        """Calls session_manager.log_session_info and model_manager.get_model_info."""
        session_manager = MagicMock()
        model_manager = MagicMock()
        model_manager.resumed_from_checkpoint = False
        model_manager.get_model_info.return_value = "Model Info"
        agent = MagicMock()
        agent.name = "test_agent"
        metrics_manager = MagicMock()
        metrics_manager.global_timestep = 0
        metrics_manager.total_episodes_completed = 0
        log_both = MagicMock()

        manager.log_run_info(
            session_manager, model_manager, agent, metrics_manager, log_both
        )

        session_manager.log_session_info.assert_called_once()
        model_manager.get_model_info.assert_called_once()
        log_both.assert_called()

    def test_handles_none_agent(self, manager):
        """Does not crash when agent is None."""
        session_manager = MagicMock()
        model_manager = MagicMock()
        model_manager.resumed_from_checkpoint = False
        model_manager.get_model_info.return_value = "Model Info"
        metrics_manager = MagicMock()
        metrics_manager.global_timestep = 0
        metrics_manager.total_episodes_completed = 0
        log_both = MagicMock()

        # Should not raise
        manager.log_run_info(
            session_manager, model_manager, None, metrics_manager, log_both
        )
        session_manager.log_session_info.assert_called_once()
