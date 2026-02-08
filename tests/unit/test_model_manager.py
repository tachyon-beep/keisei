"""Unit tests for keisei/training/model_manager.py.

Tests cover model creation, feature spec resolution, mixed precision setup,
compilation status reporting, model info retrieval, checkpoint resume logic,
and checkpoint state management.
"""

import os
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

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
from keisei.core.actor_critic_protocol import ActorCriticProtocol
from keisei.training.model_manager import ModelManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_ACTIONS = 13527
BOARD_SIZE = 9


def _make_config(
    *,
    model_type="resnet",
    input_features="core46",
    mixed_precision=False,
    enable_torch_compile=False,
    enable_compilation_benchmarking=False,
    tower_depth=1,
    tower_width=16,
    se_ratio=0.0,
    tmp_path=None,
):
    """Build a minimal AppConfig suitable for ModelManager unit tests."""
    log_dir = str(tmp_path) if tmp_path else "/tmp/test_model_manager"
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            input_channels=46,
            num_actions_total=NUM_ACTIONS,
            seed=42,
        ),
        training=TrainingConfig(
            total_timesteps=100,
            steps_per_epoch=16,
            ppo_epochs=2,
            minibatch_size=8,
            learning_rate=3e-4,
            model_type=model_type,
            input_features=input_features,
            tower_depth=tower_depth,
            tower_width=tower_width,
            se_ratio=se_ratio,
            mixed_precision=mixed_precision,
            enable_torch_compile=enable_torch_compile,
            enable_compilation_benchmarking=enable_compilation_benchmarking,
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


def _make_args(**overrides):
    """Build a minimal args namespace (simulating argparse output)."""
    defaults = {
        "input_features": None,
        "model": None,
        "tower_depth": None,
        "tower_width": None,
        "se_ratio": None,
        "resume": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _noop_logger(msg):
    """Silent logger for test usage."""
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config(tmp_path):
    """Default config: resnet, core46, compile disabled, small model."""
    return _make_config(tmp_path=tmp_path)


@pytest.fixture
def default_args():
    """Default args namespace with no overrides."""
    return _make_args()


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


@pytest.fixture
def manager(default_config, default_args, cpu_device):
    """ModelManager with default small-model config on CPU."""
    return ModelManager(
        config=default_config,
        args=default_args,
        device=cpu_device,
        logger_func=_noop_logger,
    )


# ===========================================================================
# CRITICAL: create_model()
# ===========================================================================


class TestCreateModelType:
    """create_model() creates the correct model type based on config."""

    def test_create_model_resnet_produces_valid_output(self, manager):
        """create_model() with model_type=resnet returns a model with correct forward pass."""
        model = manager.create_model()
        obs = torch.randn(1, 46, BOARD_SIZE, BOARD_SIZE)
        policy, value = model(obs)
        assert policy.shape == (1, NUM_ACTIONS)
        assert torch.isfinite(policy).all()

    def test_create_model_resnet_is_nn_module(self, manager):
        """ResNet model returned by create_model() is an nn.Module."""
        model = manager.create_model()
        assert isinstance(model, nn.Module)

    def test_create_model_resnet_type_name(self, manager):
        """ResNet model is an ActorCriticResTower instance."""
        model = manager.create_model()
        from keisei.training.models.resnet_tower import ActorCriticResTower

        assert isinstance(model, ActorCriticResTower)

    def test_create_model_dummy_type(self, tmp_path, cpu_device):
        """create_model() with model_type=dummy creates a working model."""
        config = _make_config(model_type="dummy", tmp_path=tmp_path)
        args = _make_args()
        mgr = ModelManager(config=config, args=args, device=cpu_device, logger_func=_noop_logger)
        model = mgr.create_model()
        assert isinstance(model, nn.Module)
        obs = torch.randn(1, 46, BOARD_SIZE, BOARD_SIZE)
        policy, value = model(obs)
        assert policy.shape[1] == NUM_ACTIONS


class TestCreateModelInputChannels:
    """create_model() applies the correct input channels from FeatureSpec."""

    def test_core46_produces_46_input_channels(self, manager):
        """core46 feature set leads to 46 input channels in the model."""
        model = manager.create_model()
        obs = torch.randn(1, 46, BOARD_SIZE, BOARD_SIZE)
        policy, value = model(obs)
        assert policy.shape == (1, NUM_ACTIONS)

    def test_core46_all_produces_51_input_channels(self, tmp_path, cpu_device):
        """core46+all feature set leads to 51 input channels."""
        config = _make_config(input_features="core46+all", tmp_path=tmp_path)
        args = _make_args()
        mgr = ModelManager(config=config, args=args, device=cpu_device, logger_func=_noop_logger)
        model = mgr.create_model()

        obs = torch.randn(1, 51, BOARD_SIZE, BOARD_SIZE)
        policy, value = model(obs)
        assert policy.shape == (1, NUM_ACTIONS)

    def test_obs_shape_matches_feature_spec(self, manager):
        """ModelManager.obs_shape reflects the FeatureSpec num_planes."""
        assert manager.obs_shape == (46, 9, 9)


class TestCreateModelProtocol:
    """create_model() returns a model satisfying ActorCriticProtocol."""

    def test_model_forward_returns_policy_value_pair(self, manager):
        """Model forward() returns (policy_logits, value) with correct shapes."""
        model = manager.create_model()
        obs = torch.randn(1, 46, BOARD_SIZE, BOARD_SIZE)
        policy, value = model.forward(obs)
        assert policy.shape == (1, NUM_ACTIONS)
        assert value.numel() == 1

    def test_model_get_action_and_value_returns_tuple(self, manager):
        """Model get_action_and_value() returns a result tuple."""
        model = manager.create_model()
        model.eval()
        obs = torch.randn(1, 46, BOARD_SIZE, BOARD_SIZE)
        mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
        result = model.get_action_and_value(obs, mask)
        # Should return (action, log_prob, entropy, value)
        assert len(result) >= 3

    def test_model_evaluate_actions_returns_tensors(self, manager):
        """Model evaluate_actions() returns log_probs, entropy, values."""
        model = manager.create_model()
        obs = torch.randn(2, 46, BOARD_SIZE, BOARD_SIZE)
        actions = torch.zeros(2, dtype=torch.long)
        mask = torch.ones(2, NUM_ACTIONS, dtype=torch.bool)
        log_probs, entropy, values = model.evaluate_actions(obs, actions, mask)
        assert log_probs.shape == (2,)
        assert entropy.shape == (2,)
        assert values.shape == (2,)

    def test_model_has_parameters(self, manager):
        """Model has parameters() method yielding trainable params."""
        model = manager.create_model()
        params = list(model.parameters())
        assert len(params) > 0
        assert any(p.requires_grad for p in params)

    def test_model_has_state_dict(self, manager):
        """Model has state_dict() method returning a non-empty dict."""
        model = manager.create_model()
        sd = model.state_dict()
        assert isinstance(sd, dict)
        assert len(sd) > 0

    def test_model_forward_returns_policy_and_value(self, manager):
        """Model forward() returns (policy_logits, value) tuple."""
        model = manager.create_model()
        obs = torch.randn(1, 46, BOARD_SIZE, BOARD_SIZE)
        result = model.forward(obs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        policy, value = result
        assert isinstance(policy, torch.Tensor)
        assert isinstance(value, torch.Tensor)


class TestCreateModelDevice:
    """create_model() places the model on the correct device."""

    def test_model_on_cpu(self, manager):
        """Model parameters are on CPU when device=cpu."""
        model = manager.create_model()
        for param in model.parameters():
            assert param.device == torch.device("cpu"), (
                f"Expected CPU, got {param.device}"
            )

    def test_manager_stores_model_reference(self, manager):
        """After create_model(), manager.model references the created model."""
        model = manager.create_model()
        assert manager.model is model


# ===========================================================================
# HIGH: _setup_mixed_precision()
# ===========================================================================


class TestSetupMixedPrecision:
    """_setup_mixed_precision() configures GradScaler based on config."""

    def test_scaler_none_when_mixed_precision_disabled(self, manager):
        """No GradScaler when mixed_precision=False."""
        assert manager.scaler is None
        assert manager.use_mixed_precision is False

    def test_scaler_none_on_cpu_even_when_requested(self, tmp_path, cpu_device):
        """GradScaler is None when mixed_precision=True but device is CPU."""
        config = _make_config(mixed_precision=True, tmp_path=tmp_path)
        args = _make_args()
        mgr = ModelManager(config=config, args=args, device=cpu_device, logger_func=_noop_logger)
        assert mgr.scaler is None
        assert mgr.use_mixed_precision is False

    def test_use_mixed_precision_false_on_cpu(self, tmp_path, cpu_device):
        """use_mixed_precision flag is False when device is CPU despite config=True."""
        config = _make_config(mixed_precision=True, tmp_path=tmp_path)
        args = _make_args()
        mgr = ModelManager(config=config, args=args, device=cpu_device, logger_func=_noop_logger)
        assert mgr.use_mixed_precision is False


# ===========================================================================
# HIGH: _setup_feature_spec()
# ===========================================================================


class TestSetupFeatureSpec:
    """_setup_feature_spec() resolves the correct FeatureSpec from config."""

    def test_core46_feature_spec(self, manager):
        """core46 feature set resolves to 46 planes."""
        assert manager.feature_spec.name == "core46"
        assert manager.feature_spec.num_planes == 46

    def test_core46_all_feature_spec(self, tmp_path, cpu_device):
        """core46+all feature set resolves to 51 planes."""
        config = _make_config(input_features="core46+all", tmp_path=tmp_path)
        args = _make_args()
        mgr = ModelManager(config=config, args=args, device=cpu_device, logger_func=_noop_logger)
        assert mgr.feature_spec.name == "core46+all"
        assert mgr.feature_spec.num_planes == 51

    def test_obs_shape_tuple(self, manager):
        """obs_shape is a (num_planes, 9, 9) tuple."""
        assert isinstance(manager.obs_shape, tuple)
        assert len(manager.obs_shape) == 3
        assert manager.obs_shape[1] == 9
        assert manager.obs_shape[2] == 9

    def test_dummyfeats_feature_spec(self, tmp_path, cpu_device):
        """dummyfeats feature set resolves to 46 planes."""
        config = _make_config(input_features="dummyfeats", tmp_path=tmp_path)
        args = _make_args()
        mgr = ModelManager(config=config, args=args, device=cpu_device, logger_func=_noop_logger)
        assert mgr.feature_spec.name == "dummyfeats"
        assert mgr.feature_spec.num_planes == 46


# ===========================================================================
# HIGH: get_compilation_status()
# ===========================================================================


class TestGetCompilationStatus:
    """get_compilation_status() returns the expected status dict."""

    def test_status_before_compilation_attempt(self, manager):
        """Before create_model(), compilation status shows not attempted."""
        status = manager.get_compilation_status()
        assert status["attempted"] is False
        assert "message" in status

    def test_status_after_create_model(self, manager):
        """After create_model(), compilation status is populated."""
        manager.create_model()
        status = manager.get_compilation_status()
        assert status["attempted"] is True
        assert "success" in status
        assert "compiled" in status
        assert "fallback_used" in status

    def test_status_compile_disabled_shows_success(self, manager):
        """With compile disabled, status shows success but not compiled."""
        manager.create_model()
        status = manager.get_compilation_status()
        assert status["attempted"] is True
        assert "success" in status


# ===========================================================================
# HIGH: get_model_info()
# ===========================================================================


class TestGetModelInfo:
    """get_model_info() returns complete model info dict."""

    def test_contains_model_type(self, manager):
        """Model info includes the model type."""
        info = manager.get_model_info()
        assert info["model_type"] == "resnet"

    def test_contains_input_features(self, manager):
        """Model info includes the input features name."""
        info = manager.get_model_info()
        assert info["input_features"] == "core46"

    def test_contains_tower_params(self, manager):
        """Model info includes tower depth, width, and SE ratio."""
        info = manager.get_model_info()
        assert "tower_depth" in info
        assert "tower_width" in info
        assert "se_ratio" in info

    def test_contains_obs_shape(self, manager):
        """Model info includes observation shape."""
        info = manager.get_model_info()
        assert info["obs_shape"] == (46, 9, 9)

    def test_contains_num_planes(self, manager):
        """Model info includes num_planes from feature spec."""
        info = manager.get_model_info()
        assert info["num_planes"] == 46

    def test_contains_device(self, manager):
        """Model info includes device string."""
        info = manager.get_model_info()
        assert info["device"] == "cpu"

    def test_contains_mixed_precision_flag(self, manager):
        """Model info includes use_mixed_precision flag."""
        info = manager.get_model_info()
        assert info["use_mixed_precision"] is False

    def test_contains_torch_compile_status(self, manager):
        """Model info includes torch_compile sub-dict."""
        info = manager.get_model_info()
        assert "torch_compile" in info
        assert isinstance(info["torch_compile"], dict)

    def test_contains_optimization_applied(self, manager):
        """Model info includes optimization_applied flag."""
        info = manager.get_model_info()
        assert "optimization_applied" in info

    def test_contains_benchmarking_enabled(self, manager):
        """Model info includes performance_benchmarking_enabled flag."""
        info = manager.get_model_info()
        assert "performance_benchmarking_enabled" in info


# ===========================================================================
# MEDIUM: _handle_latest_checkpoint_resume()
# ===========================================================================


class TestHandleLatestCheckpointResume:
    """_handle_latest_checkpoint_resume() finds and loads the latest checkpoint."""

    def test_no_checkpoint_returns_false(self, manager, tmp_path):
        """Returns False when no checkpoint exists in directory."""
        model_dir = str(tmp_path / "empty_model_dir")
        os.makedirs(model_dir, exist_ok=True)

        mock_agent = MagicMock()
        result = manager._handle_latest_checkpoint_resume(mock_agent, model_dir)
        assert result is False

    def test_no_checkpoint_resets_state(self, manager, tmp_path):
        """Clears checkpoint state when no checkpoint found."""
        model_dir = str(tmp_path / "empty_model_dir")
        os.makedirs(model_dir, exist_ok=True)

        # Pre-set some state to verify it gets cleared
        manager.resumed_from_checkpoint = "something"
        manager.checkpoint_data = {"key": "value"}

        mock_agent = MagicMock()
        manager._handle_latest_checkpoint_resume(mock_agent, model_dir)
        assert manager.resumed_from_checkpoint is None
        assert manager.checkpoint_data is None

    def test_loads_checkpoint_when_found(self, manager, tmp_path):
        """Loads checkpoint and returns True when checkpoint exists."""
        model_dir = str(tmp_path / "model_dir")
        os.makedirs(model_dir, exist_ok=True)
        ckpt_path = os.path.join(model_dir, "checkpoint_ts100.pth")

        # Create a real checkpoint file that find_latest_checkpoint can validate
        torch.save({"model_state_dict": {}}, ckpt_path)

        mock_agent = MagicMock()
        mock_agent.load_model.return_value = {"epoch": 10}

        result = manager._handle_latest_checkpoint_resume(mock_agent, model_dir)
        assert result is True
        assert manager.resumed_from_checkpoint is not None
        mock_agent.load_model.assert_called_once()

    def test_error_dict_stored_as_checkpoint_data(self, manager, tmp_path):
        """When load_model returns an error dict, it is stored as checkpoint_data."""
        model_dir = str(tmp_path / "model_dir")
        os.makedirs(model_dir, exist_ok=True)
        ckpt_path = os.path.join(model_dir, "checkpoint_ts100.pth")
        torch.save({"model_state_dict": {}}, ckpt_path)

        mock_agent = MagicMock()
        mock_agent.load_model.return_value = {"error": "corrupted file"}

        result = manager._handle_latest_checkpoint_resume(mock_agent, model_dir)
        assert result is True
        assert manager.checkpoint_data == {"error": "corrupted file"}


# ===========================================================================
# MEDIUM: _find_latest_checkpoint()
# ===========================================================================


class TestFindLatestCheckpoint:
    """_find_latest_checkpoint() finds checkpoint files in directory."""

    def test_finds_pth_file(self, manager, tmp_path):
        """Finds a .pth checkpoint file."""
        model_dir = str(tmp_path / "ckpt_dir")
        os.makedirs(model_dir, exist_ok=True)
        ckpt_path = os.path.join(model_dir, "model.pth")
        torch.save({"model_state_dict": {}}, ckpt_path)

        result = manager._find_latest_checkpoint(model_dir)
        assert result is not None
        assert result.endswith(".pth")

    def test_returns_none_for_empty_dir(self, manager, tmp_path):
        """Returns None when directory has no checkpoint files."""
        model_dir = str(tmp_path / "empty_dir")
        os.makedirs(model_dir, exist_ok=True)

        result = manager._find_latest_checkpoint(model_dir)
        assert result is None

    def test_returns_none_for_nonexistent_dir(self, manager, tmp_path):
        """Returns None when directory does not exist."""
        model_dir = str(tmp_path / "does_not_exist")

        result = manager._find_latest_checkpoint(model_dir)
        assert result is None


# ===========================================================================
# MEDIUM: _reset_checkpoint_state()
# ===========================================================================


class TestResetCheckpointState:
    """_reset_checkpoint_state() clears checkpoint tracking state."""

    def test_clears_resumed_from_checkpoint(self, manager):
        """Resets resumed_from_checkpoint to None."""
        manager.resumed_from_checkpoint = "/path/to/checkpoint.pth"
        manager._reset_checkpoint_state()
        assert manager.resumed_from_checkpoint is None

    def test_clears_checkpoint_data(self, manager):
        """Resets checkpoint_data to None."""
        manager.checkpoint_data = {"epoch": 10, "loss": 0.5}
        manager._reset_checkpoint_state()
        assert manager.checkpoint_data is None

    def test_idempotent_when_already_none(self, manager):
        """Calling reset when state is already None does not raise."""
        manager.resumed_from_checkpoint = None
        manager.checkpoint_data = None
        manager._reset_checkpoint_state()
        assert manager.resumed_from_checkpoint is None
        assert manager.checkpoint_data is None


# ===========================================================================
# ADDITIONAL: Args override behavior
# ===========================================================================


class TestArgsOverrideConfig:
    """ModelManager respects args overrides over config values."""

    def test_args_model_type_overrides_config(self, default_config, cpu_device):
        """args.model overrides config.training.model_type."""
        args = _make_args(model="dummy")
        mgr = ModelManager(
            config=default_config, args=args, device=cpu_device, logger_func=_noop_logger
        )
        assert mgr.model_type == "dummy"

    def test_args_tower_depth_overrides_config(self, default_config, cpu_device):
        """args.tower_depth overrides config.training.tower_depth."""
        args = _make_args(tower_depth=5)
        mgr = ModelManager(
            config=default_config, args=args, device=cpu_device, logger_func=_noop_logger
        )
        assert mgr.tower_depth == 5

    def test_args_tower_width_overrides_config(self, default_config, cpu_device):
        """args.tower_width overrides config.training.tower_width."""
        args = _make_args(tower_width=128)
        mgr = ModelManager(
            config=default_config, args=args, device=cpu_device, logger_func=_noop_logger
        )
        assert mgr.tower_width == 128

    def test_args_se_ratio_overrides_config(self, default_config, cpu_device):
        """args.se_ratio overrides config.training.se_ratio."""
        args = _make_args(se_ratio=0.5)
        mgr = ModelManager(
            config=default_config, args=args, device=cpu_device, logger_func=_noop_logger
        )
        assert mgr.se_ratio == 0.5

    def test_args_none_falls_through_to_config(self, default_config, cpu_device):
        """When args attrs are None, config values are used."""
        args = _make_args()
        mgr = ModelManager(
            config=default_config, args=args, device=cpu_device, logger_func=_noop_logger
        )
        assert mgr.model_type == default_config.training.model_type
        assert mgr.tower_depth == default_config.training.tower_depth
        assert mgr.tower_width == default_config.training.tower_width

    def test_args_se_ratio_zero_is_respected(self, default_config, cpu_device):
        """se_ratio=0 from args is not treated as falsy."""
        args = _make_args(se_ratio=0)
        mgr = ModelManager(
            config=default_config, args=args, device=cpu_device, logger_func=_noop_logger
        )
        assert mgr.se_ratio == 0


# ===========================================================================
# ADDITIONAL: handle_checkpoint_resume() dispatch
# ===========================================================================


class TestHandleCheckpointResume:
    """handle_checkpoint_resume() dispatches to the correct sub-method."""

    def test_specific_path_calls_specific_handler(self, manager, tmp_path):
        """Providing a specific resume path that does not exist returns False."""
        model_dir = str(tmp_path / "models")
        os.makedirs(model_dir, exist_ok=True)
        mock_agent = MagicMock()

        result = manager.handle_checkpoint_resume(
            mock_agent, model_dir, resume_path_override="/nonexistent/path.pth"
        )
        assert result is False

    def test_specific_path_existing_file_loads(self, manager, tmp_path):
        """Providing an existing specific path loads the checkpoint."""
        model_dir = str(tmp_path / "models")
        os.makedirs(model_dir, exist_ok=True)
        ckpt_file = os.path.join(model_dir, "specific.pth")
        torch.save({"model_state_dict": {}}, ckpt_file)

        mock_agent = MagicMock()
        mock_agent.load_model.return_value = {"epoch": 5}

        result = manager.handle_checkpoint_resume(
            mock_agent, model_dir, resume_path_override=ckpt_file
        )
        assert result is True
        assert manager.resumed_from_checkpoint == ckpt_file

    def test_latest_dispatches_to_latest_handler(self, manager, tmp_path):
        """resume_path_override=latest dispatches to _handle_latest_checkpoint_resume."""
        model_dir = str(tmp_path / "models")
        os.makedirs(model_dir, exist_ok=True)

        mock_agent = MagicMock()
        result = manager.handle_checkpoint_resume(
            mock_agent, model_dir, resume_path_override="latest"
        )
        assert result is False

    def test_none_dispatches_to_latest_handler(self, manager, tmp_path):
        """resume_path_override=None dispatches to _handle_latest_checkpoint_resume."""
        model_dir = str(tmp_path / "models")
        os.makedirs(model_dir, exist_ok=True)

        mock_agent = MagicMock()
        result = manager.handle_checkpoint_resume(
            mock_agent, model_dir, resume_path_override=None
        )
        assert result is False


# ===========================================================================
# ADDITIONAL: Model forward pass correctness
# ===========================================================================


class TestCreateModelForwardPass:
    """Verify the created model produces correct output shapes and finite values."""

    def test_single_obs_output_shapes(self, manager):
        """Single observation produces correct policy and value shapes."""
        model = manager.create_model()
        obs = torch.randn(1, 46, BOARD_SIZE, BOARD_SIZE)
        policy, value = model(obs)
        assert policy.shape == (1, NUM_ACTIONS)
        assert value.numel() == 1

    def test_batch_obs_output_shapes(self, manager):
        """Batch of observations produces correct shapes."""
        model = manager.create_model()
        batch_size = 4
        obs = torch.randn(batch_size, 46, BOARD_SIZE, BOARD_SIZE)
        policy, value = model(obs)
        assert policy.shape == (batch_size, NUM_ACTIONS)
        assert value.shape == (batch_size,)

    def test_outputs_are_finite(self, manager):
        """Model outputs are finite (no NaN or Inf)."""
        model = manager.create_model()
        obs = torch.randn(1, 46, BOARD_SIZE, BOARD_SIZE)
        policy, value = model(obs)
        assert torch.isfinite(policy).all(), "Policy logits contain NaN or Inf"
        assert torch.isfinite(value).all(), "Value contains NaN or Inf"

    def test_model_with_se_blocks(self, tmp_path, cpu_device):
        """Model with SE blocks (se_ratio > 0) produces valid outputs."""
        config = _make_config(se_ratio=0.25, tower_depth=2, tower_width=32, tmp_path=tmp_path)
        args = _make_args()
        mgr = ModelManager(config=config, args=args, device=cpu_device, logger_func=_noop_logger)
        model = mgr.create_model()

        obs = torch.randn(1, 46, BOARD_SIZE, BOARD_SIZE)
        policy, value = model(obs)
        assert policy.shape == (1, NUM_ACTIONS)
        assert torch.isfinite(policy).all()
        assert torch.isfinite(value).all()


# ===========================================================================
# ADDITIONAL: Logger function
# ===========================================================================


class TestLoggerFunction:
    """ModelManager properly uses the provided logger_func."""

    def test_default_logger_is_noop(self, default_config, default_args, cpu_device):
        """When no logger_func is provided, a no-op lambda is used."""
        mgr = ModelManager(config=default_config, args=default_args, device=cpu_device)
        mgr.logger_func("test message")

    def test_custom_logger_receives_messages(self, default_config, default_args, cpu_device):
        """Custom logger_func receives log messages."""
        messages = []
        mgr = ModelManager(
            config=default_config,
            args=default_args,
            device=cpu_device,
            logger_func=messages.append,
        )
        assert len(messages) > 0
