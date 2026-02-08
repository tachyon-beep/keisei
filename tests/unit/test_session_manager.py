"""Tests for SessionManager: session lifecycle management for training runs.

Covers directory setup, seeding, run name handling, config serialization,
WandB integration (mocked), session logging, and session summary.
"""

import json
import os
import types
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path, wandb_enabled=False, seed=42, run_name=None):
    """Build a minimal AppConfig for SessionManager tests."""
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            input_channels=46,
            num_actions_total=13527,
            seed=seed,
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
            run_name=run_name,
        ),
        wandb=WandBConfig(enabled=wandb_enabled),
        parallel=ParallelConfig(enabled=False),
        display=DisplayConfig(display_moves=False),
        webui=WebUIConfig(enabled=False),
    )


def _make_args(resume=None, run_name=None):
    """Build a minimal args namespace mimicking CLI arguments."""
    return types.SimpleNamespace(resume=resume, config=None, run_name=run_name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    """Minimal AppConfig for testing."""
    return _make_config(tmp_path)


@pytest.fixture
def args():
    """Minimal CLI args namespace."""
    return _make_args()


@pytest.fixture
def session_manager(tmp_path):
    """SessionManager with directories already set up."""
    from keisei.training.session_manager import SessionManager

    cfg = _make_config(tmp_path)
    sm = SessionManager(cfg, _make_args(), run_name="test_run")
    sm.setup_directories()
    return sm


# ---------------------------------------------------------------------------
# Tests: Run Name
# ---------------------------------------------------------------------------


class TestRunName:
    """Run name determination and sanitization."""

    def test_run_name_explicit_overrides_all(self, tmp_path):
        """Explicit run_name parameter takes priority over args and config."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path, run_name="config_name")
        sm = SessionManager(cfg, _make_args(run_name="args_name"), run_name="explicit_name")
        assert sm.run_name == "explicit_name"

    def test_run_name_from_args_overrides_config(self, tmp_path):
        """CLI args.run_name takes priority over config.logging.run_name."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path, run_name="config_name")
        sm = SessionManager(cfg, _make_args(run_name="args_name"))
        assert sm.run_name == "args_name"

    def test_run_name_from_config_when_no_explicit_or_args(self, tmp_path):
        """config.logging.run_name is used when no explicit or args name."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path, run_name="config_name")
        sm = SessionManager(cfg, _make_args())
        assert sm.run_name == "config_name"

    def test_run_name_auto_generated_when_none_provided(self, tmp_path):
        """Auto-generated name is used when no explicit, args, or config name."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args())
        assert sm.run_name.startswith("keisei")
        assert len(sm.run_name) > 0

    def test_run_name_sanitizes_path_traversal(self, tmp_path):
        """Path traversal characters in run name are sanitized."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="../../../etc/passwd")
        assert ".." not in sm.run_name
        assert "/" not in sm.run_name

    def test_run_name_sanitizes_special_characters(self, tmp_path):
        """Special characters in run name are replaced with underscores."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="run name with spaces!")
        assert " " not in sm.run_name
        assert "!" not in sm.run_name

    def test_run_name_empty_becomes_unnamed_run(self, tmp_path):
        """An empty or all-invalid run name becomes unnamed_run."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="...")
        assert sm.run_name == "unnamed_run"


# ---------------------------------------------------------------------------
# Tests: setup_directories
# ---------------------------------------------------------------------------


class TestSetupDirectories:
    """Directory structure creation."""

    def test_setup_directories_creates_run_artifact_dir(self, tmp_path):
        """setup_directories creates the run artifact directory."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        dirs = sm.setup_directories()
        assert os.path.isdir(dirs["run_artifact_dir"])

    def test_setup_directories_returns_expected_keys(self, tmp_path):
        """Returned dict contains all required keys."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        dirs = sm.setup_directories()
        expected_keys = {
            "run_artifact_dir",
            "model_dir",
            "log_file_path",
            "eval_log_file_path",
        }
        assert set(dirs.keys()) == expected_keys

    def test_setup_directories_sets_model_dir_property(self, tmp_path):
        """model_dir property returns correct path after setup."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        assert "test_run" in sm.model_dir
        assert sm.model_dir.startswith(str(tmp_path))

    def test_setup_directories_sets_run_artifact_dir_property(self, tmp_path):
        """run_artifact_dir property returns correct path after setup."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        assert "test_run" in sm.run_artifact_dir

    def test_setup_directories_handles_existing_directory(self, tmp_path):
        """Calling setup_directories when directory exists does not raise."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        expected_dir = os.path.join(str(tmp_path / "models"), "test_run")
        os.makedirs(expected_dir, exist_ok=True)
        dirs = sm.setup_directories()
        assert os.path.isdir(dirs["run_artifact_dir"])

    def test_setup_directories_log_file_path_contains_run_name(self, tmp_path):
        """Log file path is inside the run artifact directory."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        assert "test_run" in sm.log_file_path

    def test_setup_directories_eval_log_file_path_set(self, tmp_path):
        """eval_log_file_path is inside the run artifact directory."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        assert "test_run" in sm.eval_log_file_path


# ---------------------------------------------------------------------------
# Tests: Property access before setup
# ---------------------------------------------------------------------------


class TestPropertyAccessBeforeSetup:
    """Properties raise RuntimeError before setup_directories() is called."""

    def test_model_dir_raises_before_setup(self, tmp_path):
        """Accessing model_dir before setup_directories raises RuntimeError."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        with pytest.raises(RuntimeError, match="Directories not yet set up"):
            _ = sm.model_dir

    def test_run_artifact_dir_raises_before_setup(self, tmp_path):
        """Accessing run_artifact_dir before setup raises RuntimeError."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        with pytest.raises(RuntimeError, match="Directories not yet set up"):
            _ = sm.run_artifact_dir

    def test_log_file_path_raises_before_setup(self, tmp_path):
        """Accessing log_file_path before setup raises RuntimeError."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        with pytest.raises(RuntimeError, match="Directories not yet set up"):
            _ = sm.log_file_path

    def test_eval_log_file_path_raises_before_setup(self, tmp_path):
        """Accessing eval_log_file_path before setup raises RuntimeError."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        with pytest.raises(RuntimeError, match="Directories not yet set up"):
            _ = sm.eval_log_file_path

    def test_is_wandb_active_raises_before_setup(self, tmp_path):
        """Accessing is_wandb_active before setup_wandb raises RuntimeError."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        with pytest.raises(RuntimeError, match="WandB not yet initialized"):
            _ = sm.is_wandb_active


# ---------------------------------------------------------------------------
# Tests: setup_seeding
# ---------------------------------------------------------------------------


class TestSetupSeeding:
    """Random seed setup and reproducibility."""

    def test_setup_seeding_produces_reproducible_torch_output(self, tmp_path):
        """Same seed produces identical torch.rand output."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path, seed=12345)
        sm = SessionManager(cfg, _make_args(), run_name="seed_test")
        sm.setup_seeding()
        result_a = torch.rand(5)

        sm.setup_seeding()
        result_b = torch.rand(5)

        assert torch.allclose(result_a, result_b)

    def test_setup_seeding_different_seeds_produce_different_output(self, tmp_path):
        """Different seeds produce different torch.rand output."""
        from keisei.training.session_manager import SessionManager

        cfg1 = _make_config(tmp_path, seed=111)
        sm1 = SessionManager(cfg1, _make_args(), run_name="seed_test_1")
        sm1.setup_seeding()
        result_1 = torch.rand(10)

        cfg2 = _make_config(tmp_path, seed=222)
        sm2 = SessionManager(cfg2, _make_args(), run_name="seed_test_2")
        sm2.setup_seeding()
        result_2 = torch.rand(10)

        assert not torch.allclose(result_1, result_2)


# ---------------------------------------------------------------------------
# Tests: save_effective_config
# ---------------------------------------------------------------------------


class TestSaveEffectiveConfig:
    """Configuration serialization and persistence."""

    def test_save_effective_config_creates_file(self, session_manager):
        """save_effective_config creates effective_config.json in run dir."""
        session_manager.save_effective_config()
        config_path = os.path.join(
            session_manager.run_artifact_dir, "effective_config.json"
        )
        assert os.path.exists(config_path)

    def test_save_effective_config_is_valid_json(self, session_manager):
        """Saved config file contains valid JSON."""
        session_manager.save_effective_config()
        config_path = os.path.join(
            session_manager.run_artifact_dir, "effective_config.json"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_save_effective_config_contains_training_section(self, session_manager):
        """Saved config includes the training section with correct values."""
        session_manager.save_effective_config()
        config_path = os.path.join(
            session_manager.run_artifact_dir, "effective_config.json"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "training" in data
        assert data["training"]["learning_rate"] == pytest.approx(3e-4)

    def test_save_effective_config_contains_env_section(self, session_manager):
        """Saved config includes the env section with correct seed."""
        session_manager.save_effective_config()
        config_path = os.path.join(
            session_manager.run_artifact_dir, "effective_config.json"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "env" in data
        assert data["env"]["seed"] == 42

    def test_save_effective_config_raises_before_directory_setup(self, tmp_path):
        """Saving config before directories are set up raises RuntimeError."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        with pytest.raises(RuntimeError, match="Directories must be set up"):
            sm.save_effective_config()


# ---------------------------------------------------------------------------
# Tests: log_session_start
# ---------------------------------------------------------------------------


class TestLogSessionStart:
    """Session start logging."""

    def test_log_session_start_writes_to_log_file(self, session_manager):
        """log_session_start appends a SESSION START line to the log file."""
        session_manager.log_session_start()
        with open(session_manager.log_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "SESSION START" in content
        assert "test_run" in content

    def test_log_session_start_appends_without_overwriting(self, session_manager):
        """Calling log_session_start twice appends, not overwrites."""
        session_manager.log_session_start()
        session_manager.log_session_start()
        with open(session_manager.log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        session_start_lines = [line for line in lines if "SESSION START" in line]
        assert len(session_start_lines) == 2

    def test_log_session_start_raises_before_directory_setup(self, tmp_path):
        """Logging session start before directories raises RuntimeError."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        with pytest.raises(RuntimeError, match="Directories must be set up"):
            sm.log_session_start()


# ---------------------------------------------------------------------------
# Tests: log_session_info
# ---------------------------------------------------------------------------


class TestLogSessionInfo:
    """Session info logging with various parameters."""

    def test_log_session_info_logs_run_name(self, session_manager):
        """log_session_info includes the run name in logged output."""
        logged = []
        session_manager._is_wandb_active = False
        session_manager.log_session_info(logger_func=logged.append)
        full_output = " ".join(logged)
        assert "test_run" in full_output

    def test_log_session_info_logs_device(self, session_manager):
        """log_session_info includes the device type."""
        logged = []
        session_manager._is_wandb_active = False
        session_manager.log_session_info(logger_func=logged.append)
        full_output = " ".join(logged)
        assert "cpu" in full_output

    def test_log_session_info_fresh_training(self, session_manager):
        """log_session_info says Starting fresh training when timestep is 0."""
        logged = []
        session_manager._is_wandb_active = False
        session_manager.log_session_info(
            logger_func=logged.append, global_timestep=0
        )
        full_output = " ".join(logged)
        assert "Starting fresh training" in full_output

    def test_log_session_info_resumed_training(self, session_manager):
        """log_session_info reports resumed state when global_timestep > 0."""
        logged = []
        session_manager._is_wandb_active = False
        session_manager.log_session_info(
            logger_func=logged.append,
            global_timestep=100,
            total_episodes_completed=10,
            resumed_from_checkpoint="/some/checkpoint.pt",
        )
        full_output = " ".join(logged)
        assert "Resuming from timestep 100" in full_output
        assert "checkpoint" in full_output.lower()

    def test_log_session_info_with_agent_info(self, session_manager):
        """log_session_info includes agent info when provided."""
        logged = []
        session_manager._is_wandb_active = False
        agent_info = {"type": "PPO", "name": "TestAgent"}
        session_manager.log_session_info(
            logger_func=logged.append, agent_info=agent_info
        )
        full_output = " ".join(logged)
        assert "PPO" in full_output
        assert "TestAgent" in full_output


# ---------------------------------------------------------------------------
# Tests: setup_wandb
# ---------------------------------------------------------------------------


class TestSetupWandB:
    """WandB initialization with mocked wandb module."""

    def test_setup_wandb_returns_false_when_disabled(self, tmp_path):
        """setup_wandb returns False when wandb is disabled in config."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path, wandb_enabled=False)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        result = sm.setup_wandb()
        assert result is False

    def test_setup_wandb_sets_is_wandb_active_false_when_disabled(self, tmp_path):
        """is_wandb_active is False after setup_wandb with disabled config."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path, wandb_enabled=False)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        sm.setup_wandb()
        assert sm.is_wandb_active is False

    def test_setup_wandb_raises_before_directory_setup(self, tmp_path):
        """Calling setup_wandb before directories raises RuntimeError."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path, wandb_enabled=False)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        with pytest.raises(RuntimeError, match="Directories must be set up"):
            sm.setup_wandb()

    def test_setup_wandb_handles_exception_gracefully(self, tmp_path):
        """setup_wandb returns False when wandb init raises an error."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path, wandb_enabled=True)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        with patch(
            "keisei.training.session_manager.utils.setup_wandb",
            side_effect=ImportError("no wandb"),
        ):
            result = sm.setup_wandb()
        assert result is False
        assert sm.is_wandb_active is False


# ---------------------------------------------------------------------------
# Tests: is_wandb_active property
# ---------------------------------------------------------------------------


class TestIsWandBActive:
    """is_wandb_active property reflects actual state."""

    def test_is_wandb_active_false_after_disabled_setup(self, tmp_path):
        """is_wandb_active is False when wandb is disabled."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path, wandb_enabled=False)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        sm.setup_wandb()
        assert sm.is_wandb_active is False

    def test_is_wandb_active_true_when_wandb_succeeds(self, tmp_path):
        """is_wandb_active is True when wandb init succeeds."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path, wandb_enabled=True)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        with patch(
            "keisei.training.session_manager.utils.setup_wandb", return_value=True
        ):
            sm.setup_wandb()
        assert sm.is_wandb_active is True


# ---------------------------------------------------------------------------
# Tests: finalize_session
# ---------------------------------------------------------------------------


class TestFinalizeSession:
    """Session finalization and cleanup."""

    def test_finalize_session_no_wandb_does_not_crash(self, tmp_path):
        """finalize_session runs without error when wandb is inactive."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path, wandb_enabled=False)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        sm.setup_wandb()
        sm.finalize_session()

    def test_finalize_session_with_active_wandb_calls_finish(self, tmp_path):
        """finalize_session calls wandb.finish() when wandb is active."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path, wandb_enabled=True)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        sm._is_wandb_active = True

        mock_run = MagicMock()
        mock_finish = MagicMock()
        with patch("keisei.training.session_manager.wandb") as mock_wandb:
            mock_wandb.run = mock_run
            mock_wandb.finish = mock_finish
            sm.finalize_session()
        mock_finish.assert_called()


# ---------------------------------------------------------------------------
# Tests: get_session_summary
# ---------------------------------------------------------------------------


class TestGetSessionSummary:
    """Session summary dictionary."""

    def test_get_session_summary_returns_dict(self, session_manager):
        """get_session_summary returns a dict."""
        summary = session_manager.get_session_summary()
        assert isinstance(summary, dict)

    def test_get_session_summary_contains_run_name(self, session_manager):
        """Summary includes the run name."""
        summary = session_manager.get_session_summary()
        assert summary["run_name"] == "test_run"

    def test_get_session_summary_contains_directory_paths(self, session_manager):
        """Summary includes directory paths after setup."""
        summary = session_manager.get_session_summary()
        assert summary["run_artifact_dir"] is not None
        assert summary["model_dir"] is not None
        assert summary["log_file_path"] is not None
        assert summary["eval_log_file_path"] is not None

    def test_get_session_summary_contains_seed(self, session_manager):
        """Summary includes the seed value."""
        summary = session_manager.get_session_summary()
        assert summary["seed"] == 42

    def test_get_session_summary_contains_device(self, session_manager):
        """Summary includes the device."""
        summary = session_manager.get_session_summary()
        assert summary["device"] == "cpu"

    def test_get_session_summary_wandb_state_none_before_wandb_setup(self, tmp_path):
        """is_wandb_active in summary is None before setup_wandb."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        summary = sm.get_session_summary()
        assert summary["is_wandb_active"] is None

    def test_get_session_summary_has_all_expected_keys(self, session_manager):
        """Summary dictionary contains all expected keys."""
        summary = session_manager.get_session_summary()
        expected_keys = {
            "run_name",
            "run_artifact_dir",
            "model_dir",
            "log_file_path",
            "eval_log_file_path",
            "is_wandb_active",
            "seed",
            "device",
        }
        assert set(summary.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: log_evaluation_metrics
# ---------------------------------------------------------------------------


class TestLogEvaluationMetrics:
    """Evaluation metrics logging with mocked wandb."""

    def test_log_evaluation_metrics_none_result_does_not_crash(self, tmp_path):
        """Passing None as result does not raise when wandb is inactive."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path, wandb_enabled=False)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        sm._is_wandb_active = False
        sm.log_evaluation_metrics(None, step=0)

    def test_log_evaluation_metrics_with_summary_stats(self, tmp_path):
        """Metrics from summary_stats are logged to wandb."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        sm._is_wandb_active = True

        result = types.SimpleNamespace(
            summary_stats=types.SimpleNamespace(
                win_rate=0.6,
                loss_rate=0.3,
                draw_rate=0.1,
                total_games=10,
                avg_game_length=50.0,
                avg_rewards=1.5,
            ),
        )

        mock_run = MagicMock()
        mock_log = MagicMock()
        with patch("keisei.training.session_manager.wandb") as mock_wandb:
            mock_wandb.run = mock_run
            mock_wandb.log = mock_log
            sm.log_evaluation_metrics(result, step=100)
        mock_log.assert_called_once()
        logged_metrics = mock_log.call_args[0][0]
        assert logged_metrics["evaluation/win_rate"] == 0.6
        assert logged_metrics["evaluation/loss_rate"] == 0.3

    def test_log_evaluation_metrics_skipped_when_wandb_inactive(self, tmp_path):
        """No wandb.log call when wandb is inactive."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        sm._is_wandb_active = False

        result = types.SimpleNamespace(
            summary_stats=types.SimpleNamespace(win_rate=0.6),
        )

        with patch("keisei.training.session_manager.wandb") as mock_wandb:
            mock_wandb.run = None
            sm.log_evaluation_metrics(result, step=100)
            mock_wandb.log.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: log_evaluation_performance
# ---------------------------------------------------------------------------


class TestLogEvaluationPerformance:
    """Evaluation performance logging with mocked wandb."""

    def test_log_evaluation_performance_prefixes_metrics(self, tmp_path):
        """Performance metrics are prefixed with evaluation/performance/."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        sm._is_wandb_active = True

        metrics = {"latency_ms": 50.0, "memory_overhead_mb": 100.0}

        mock_run = MagicMock()
        mock_log = MagicMock()
        with patch("keisei.training.session_manager.wandb") as mock_wandb:
            mock_wandb.run = mock_run
            mock_wandb.log = mock_log
            sm.log_evaluation_performance(metrics, step=200)
        mock_log.assert_called_once()
        logged = mock_log.call_args[0][0]
        assert "evaluation/performance/latency_ms" in logged
        assert "evaluation/performance/memory_overhead_mb" in logged

    def test_log_evaluation_performance_skipped_when_inactive(self, tmp_path):
        """No wandb.log when wandb is inactive."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        sm._is_wandb_active = False

        with patch("keisei.training.session_manager.wandb") as mock_wandb:
            mock_wandb.run = None
            sm.log_evaluation_performance({"latency_ms": 50.0}, step=200)
            mock_wandb.log.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: log_evaluation_sla_status
# ---------------------------------------------------------------------------


class TestLogEvaluationSlaStatus:
    """Evaluation SLA status logging."""

    def test_log_sla_status_logs_pass_status(self, tmp_path):
        """SLA pass status is logged correctly."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        sm._is_wandb_active = True

        mock_run = MagicMock()
        mock_log = MagicMock()
        with patch("keisei.training.session_manager.wandb") as mock_wandb:
            mock_wandb.run = mock_run
            mock_wandb.log = mock_log
            sm.log_evaluation_sla_status(True, [], step=300)
        mock_log.assert_called_once()
        logged = mock_log.call_args[0][0]
        assert logged["evaluation/sla/passed"] is True
        assert logged["evaluation/sla/violation_count"] == 0

    def test_log_sla_status_logs_violations(self, tmp_path):
        """SLA violations are logged with correct count."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        sm._is_wandb_active = True

        violations = ["latency=high", "memory=exceeded"]
        mock_run = MagicMock()
        mock_log = MagicMock()
        with patch("keisei.training.session_manager.wandb") as mock_wandb:
            mock_wandb.run = mock_run
            mock_wandb.log = mock_log
            sm.log_evaluation_sla_status(False, violations, step=300)
        mock_log.assert_called_once()
        logged = mock_log.call_args[0][0]
        assert logged["evaluation/sla/passed"] is False
        assert logged["evaluation/sla/violation_count"] == 2


# ---------------------------------------------------------------------------
# Tests: setup_evaluation_logging
# ---------------------------------------------------------------------------


class TestSetupEvaluationLogging:
    """Evaluation logging setup with mocked wandb."""

    def test_setup_evaluation_logging_updates_wandb_config(self, tmp_path):
        """setup_evaluation_logging updates wandb.config with eval params."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        sm._is_wandb_active = True

        eval_config = types.SimpleNamespace(
            strategy="single_opponent",
            num_games=20,
            max_concurrent_games=4,
            opponent_type="random",
            enable_periodic_evaluation=True,
            evaluation_interval_timesteps=50000,
            enable_performance_monitoring=True,
        )

        mock_run = MagicMock()
        mock_config = MagicMock()
        with patch("keisei.training.session_manager.wandb") as mock_wandb:
            mock_wandb.run = mock_run
            mock_wandb.config = mock_config
            sm.setup_evaluation_logging(eval_config)
        mock_config.update.assert_called()

    def test_setup_evaluation_logging_skipped_when_inactive(self, tmp_path):
        """No wandb calls when wandb is inactive."""
        from keisei.training.session_manager import SessionManager

        cfg = _make_config(tmp_path)
        sm = SessionManager(cfg, _make_args(), run_name="test_run")
        sm.setup_directories()
        sm._is_wandb_active = False

        eval_config = types.SimpleNamespace(
            strategy="single_opponent",
            num_games=20,
            max_concurrent_games=4,
            opponent_type="random",
            enable_periodic_evaluation=True,
            evaluation_interval_timesteps=50000,
            enable_performance_monitoring=True,
        )

        with patch("keisei.training.session_manager.wandb") as mock_wandb:
            mock_wandb.run = None
            sm.setup_evaluation_logging(eval_config)
            mock_wandb.config.update.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _sanitize_run_name static method
# ---------------------------------------------------------------------------


class TestSanitizeRunName:
    """Static method for run name sanitization."""

    def test_sanitize_preserves_valid_name(self):
        """Valid name with alphanumeric, hyphens, underscores passes through."""
        from keisei.training.session_manager import SessionManager

        assert SessionManager._sanitize_run_name("my-run_v1.0") == "my-run_v1.0"

    def test_sanitize_strips_path_separators(self):
        """Path separators are stripped by os.path.basename."""
        from keisei.training.session_manager import SessionManager

        result = SessionManager._sanitize_run_name("/path/to/run_name")
        assert "/" not in result
        assert result == "run_name"

    def test_sanitize_replaces_special_chars(self):
        """Special characters are replaced with underscores."""
        from keisei.training.session_manager import SessionManager

        result = SessionManager._sanitize_run_name("run@name#v1")
        assert "@" not in result
        assert "#" not in result

    def test_sanitize_collapses_multiple_dots(self):
        """Multiple consecutive dots are collapsed to one."""
        from keisei.training.session_manager import SessionManager

        result = SessionManager._sanitize_run_name("run...name")
        assert "..." not in result

    def test_sanitize_strips_leading_dots(self):
        """Leading dots are stripped."""
        from keisei.training.session_manager import SessionManager

        result = SessionManager._sanitize_run_name(".hidden_run")
        assert not result.startswith(".")

    def test_sanitize_empty_becomes_unnamed_run(self):
        """Empty string after sanitization becomes unnamed_run."""
        from keisei.training.session_manager import SessionManager

        result = SessionManager._sanitize_run_name("")
        assert result == "unnamed_run"
