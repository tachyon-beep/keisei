"""
Unit tests for ModelManager initialization and basic utilities.
"""

import tempfile
from unittest.mock import Mock, patch

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
)
from keisei.training.model_manager import ModelManager


class MockArgs:
    """Mock command-line arguments for testing."""

    def __init__(self, **kwargs):
        self.resume = kwargs.get("resume", None)
        self.input_features = kwargs.get("input_features", None)
        self.model = kwargs.get("model", None)
        self.tower_depth = kwargs.get("tower_depth", None)
        self.tower_width = kwargs.get("tower_width", None)
        self.se_ratio = kwargs.get("se_ratio", None)


@pytest.fixture
def mock_config():
    """Create a mock AppConfig for testing."""
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            num_actions_total=1,
            input_channels=1,
            seed=42,
            max_moves_per_game=500,
        ),
        training=TrainingConfig(
            total_timesteps=1,
            steps_per_epoch=1,
            ppo_epochs=1,
            minibatch_size=2,  # Changed from 1 to 2
            learning_rate=1e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=1,
            refresh_per_second=1,
            enable_spinner=False,
            input_features="core",
            tower_depth=1,
            tower_width=1,
            se_ratio=0.1,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=1,
            evaluation_interval_timesteps=1,
            weight_decay=0.0,
            normalize_advantages=True,
            enable_value_clipping=False,  # Added
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            enable_periodic_evaluation=True,
            evaluation_interval_timesteps=1,  # Moved up
            strategy="single_opponent",  # Added
            num_games=1,
            max_concurrent_games=4,  # Added
            timeout_per_game=None,  # Added
            opponent_type="random",
            max_moves_per_game=500,
            randomize_positions=True,  # Added
            random_seed=None,  # Added
            save_games=True,  # Added
            save_path=None,  # Added
            log_file_path_eval="test_eval.log",
            log_level="INFO",  # Added
            wandb_log_eval=False,
            update_elo=True,  # Added
            elo_registry_path="elo_ratings.json",  # Added
            agent_id=None,  # Added
            opponent_id=None,  # Added
            previous_model_pool_size=5,  # Added
            enable_in_memory_evaluation=True,  # Added
            model_weight_cache_size=5,  # Added
            enable_parallel_execution=True,  # Added
            process_restart_threshold=100,  # Added
            temp_agent_device="cpu",  # Added
            clear_cache_after_evaluation=True,  # Added
        ),
        logging=LoggingConfig(
            log_file="test.log", model_dir=tempfile.gettempdir(), run_name=None
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1,
            watch_log_type="all",
            log_model_artifact=False,
        ),
        parallel=ParallelConfig(
            enabled=False,
            num_workers=4,
            batch_size=32,
            sync_interval=100,
            compression_enabled=True,
            timeout_seconds=10.0,
            max_queue_size=1000,
            worker_seed_offset=1000,
        ),
        display=DisplayConfig(
            enable_board_display=True,  # Added
            enable_trend_visualization=True,  # Added
            enable_elo_ratings=True,  # Added
            enable_enhanced_layout=True,  # Added
            display_moves=False,
            turn_tick=0.0,
            board_unicode_pieces=True,  # Added
            board_cell_width=5,  # Added
            board_cell_height=3,  # Added
            board_highlight_last_move=True,  # Added
            sparkline_width=15,  # Added
            trend_history_length=100,  # Added
            elo_initial_rating=1500.0,  # Added
            elo_k_factor=32.0,  # Added
            dashboard_height_ratio=2,  # Added
            progress_bar_height=4,  # Added
            show_text_moves=True,  # Added
            move_list_length=10,  # Added
            moves_latest_top=True,  # Added
            moves_flash_ms=500,  # Added
            show_moves_trend=True,  # Added
            show_completion_rate=True,  # Added
            show_enhanced_win_rates=True,  # Added
            show_turns_trend=True,  # Added
            metrics_window_size=100,  # Added
            trend_smoothing_factor=0.1,  # Added
            metrics_panel_height=6,  # Added
            enable_trendlines=True,  # Added
            log_layer_keyword_filters=["stem", "policy_head", "value_head"],  # Added
        ),
    )


@pytest.fixture
def mock_args():
    """Create mock command-line arguments."""
    return MockArgs()


@pytest.fixture
def device():
    """Create a test device."""
    return torch.device("cpu")


@pytest.fixture
def logger_func():
    """Create a mock logger function."""
    return Mock()


@pytest.fixture
def minimal_model_manager_config():
    """Create a minimal AppConfig for ModelManager testing."""
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            num_actions_total=13527,
            input_channels=46,
            seed=42,
            max_moves_per_game=500,
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=64,
            ppo_epochs=2,
            minibatch_size=32,
            learning_rate=3e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=False,
            input_features="core46",
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=10000,
            evaluation_interval_timesteps=50000,
            weight_decay=0.0,
            normalize_advantages=True,
            enable_value_clipping=False,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            enable_periodic_evaluation=False,
            strategy="single_opponent",
            num_games=1,
            max_concurrent_games=1,
            timeout_per_game=None,
            opponent_type="random",
            evaluation_interval_timesteps=50000,
            max_moves_per_game=500,
            randomize_positions=False,
            random_seed=None,
            save_games=False,
            save_path=None,
            log_file_path_eval="eval_log.txt",
            log_level="INFO",
            wandb_log_eval=False,
            update_elo=False,
            elo_registry_path=None,
            agent_id=None,
            opponent_id=None,
            previous_model_pool_size=1,
            enable_in_memory_evaluation=False,
            model_weight_cache_size=1,
            enable_parallel_execution=False,
            process_restart_threshold=100,
            temp_agent_device="cpu",
            clear_cache_after_evaluation=True,
        ),
        logging=LoggingConfig(
            log_file="test.log",
            model_dir="/tmp/test_models",
            run_name=None,
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test-project",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
            log_model_artifact=False,
        ),
        parallel=ParallelConfig(
            enabled=False,
            num_workers=1,
            batch_size=32,
            sync_interval=100,
            compression_enabled=False,
            timeout_seconds=10.0,
            max_queue_size=1000,
            worker_seed_offset=1000,
        ),
        display=DisplayConfig(
            enable_board_display=False,
            enable_trend_visualization=False,
            enable_elo_ratings=False,
            enable_enhanced_layout=False,
            display_moves=False,
            turn_tick=0.0,
            board_unicode_pieces=False,
            board_cell_width=5,
            board_cell_height=3,
            board_highlight_last_move=False,
            sparkline_width=15,
            trend_history_length=100,
            elo_initial_rating=1500.0,
            elo_k_factor=32.0,
            dashboard_height_ratio=2,
            progress_bar_height=4,
            show_text_moves=False,
            move_list_length=10,
            moves_latest_top=True,
            moves_flash_ms=0,
            show_moves_trend=False,
            show_completion_rate=False,
            show_enhanced_win_rates=False,
            show_turns_trend=False,
            metrics_window_size=100,
            trend_smoothing_factor=0.1,
            metrics_panel_height=6,
            enable_trendlines=False,
            log_layer_keyword_filters=["stem", "policy_head", "value_head"],
        ),
    )


@pytest.fixture
def mock_feature_spec():
    """Create a consistent mock feature spec."""
    mock_spec = Mock()
    mock_spec.num_planes = 46
    return mock_spec


class TestModelManagerInitialization:
    """Test ModelManager initialization and configuration."""

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_initialization_success(
        self,
        mock_model_factory,
        mock_feature_specs,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test successful ModelManager initialization."""
        # Setup mocks
        mock_feature_specs.__getitem__.return_value = Mock(num_planes=46)
        mock_model_factory.return_value = Mock()

        # Create ModelManager
        manager = ModelManager(
            minimal_model_manager_config, mock_args, device, logger_func
        )

        # Verify initialization
        assert manager.config == minimal_model_manager_config
        assert manager.args == mock_args
        assert manager.device == device
        assert manager.logger_func == logger_func
        assert manager.model is None  # Model not created yet
        assert manager.resumed_from_checkpoint is None
        assert manager.checkpoint_data is None

        # Verify feature setup
        assert (
            manager.input_features
            == minimal_model_manager_config.training.input_features
        )
        assert manager.obs_shape == (46, 9, 9)

        # Verify mixed precision setup for CPU
        assert not manager.use_mixed_precision
        assert manager.scaler is None

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_initialization_with_args_override(
        self,
        mock_model_factory,
        mock_feature_specs,
        minimal_model_manager_config,
        device,
        logger_func,
    ):
        """Test ModelManager initialization with args override."""
        # Setup mocks
        mock_feature_specs.__getitem__.return_value = Mock(num_planes=20)
        mock_model_factory.return_value = Mock()

        # Create args with overrides
        args_with_overrides = MockArgs(
            input_features="custom",
            model="cnn",
            tower_depth=8,
            tower_width=128,
            se_ratio=0.1,
        )

        # Create ModelManager
        manager = ModelManager(
            minimal_model_manager_config, args_with_overrides, device, logger_func
        )

        # Verify args override config values
        assert manager.input_features == "custom"
        assert manager.model_type == "cnn"
        assert manager.tower_depth == 8
        assert manager.tower_width == 128
        assert (
            abs(manager.se_ratio - 0.1) < 1e-9
        )  # Use approximate comparison for float
        assert manager.obs_shape == (20, 9, 9)

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_falsy_args_override_not_ignored(
        self,
        mock_model_factory,
        mock_feature_specs,
        minimal_model_manager_config,
        device,
        logger_func,
    ):
        """Regression test for P2-08: se_ratio=0 must not fall through to config default.

        The `getattr(args, ...) or config_default` pattern treats valid falsy
        values (0, 0.0) as "not provided" and silently uses the config default.
        """
        mock_feature_specs.__getitem__.return_value = Mock(num_planes=46)
        mock_model_factory.return_value = Mock()

        # se_ratio=0 means "disable SE blocks" — a valid architectural choice
        args_with_zero = MockArgs(se_ratio=0)

        manager = ModelManager(
            minimal_model_manager_config, args_with_zero, device, logger_func
        )

        # Must be 0, not the config default (0.25)
        assert manager.se_ratio == 0

    @patch("keisei.training.model_manager.GradScaler")
    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_mixed_precision_cuda_enabled(
        self,
        mock_model_factory,
        mock_feature_specs,
        mock_grad_scaler,
        minimal_model_manager_config,
        mock_args,
        logger_func,
    ):
        """Test mixed precision setup when CUDA is available."""
        # Setup mocks
        mock_feature_specs.__getitem__.return_value = Mock(num_planes=46)
        mock_model_factory.return_value = Mock()
        mock_scaler_instance = Mock()
        mock_grad_scaler.return_value = mock_scaler_instance

        # Enable mixed precision in config
        minimal_model_manager_config.training.mixed_precision = True

        # Use CUDA device
        cuda_device = torch.device("cuda")

        # Create ModelManager
        manager = ModelManager(
            minimal_model_manager_config, mock_args, cuda_device, logger_func
        )

        # Verify mixed precision is enabled
        assert manager.use_mixed_precision is True
        assert manager.scaler == mock_scaler_instance
        mock_grad_scaler.assert_called_once()

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_mixed_precision_cpu_warning(
        self,
        mock_model_factory,
        mock_feature_specs,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test mixed precision warning when requested on CPU."""
        # Setup mocks
        mock_feature_specs.__getitem__.return_value = Mock(num_planes=46)
        mock_model_factory.return_value = Mock()

        # Enable mixed precision in config (but device is CPU)
        minimal_model_manager_config.training.mixed_precision = True

        # Create ModelManager
        manager = ModelManager(
            minimal_model_manager_config, mock_args, device, logger_func
        )

        # Verify mixed precision is disabled with warning
        assert manager.use_mixed_precision is False
        assert manager.scaler is None

        # Verify warning was logged (check all calls for the mixed precision warning)
        logger_func.assert_called()
        # Check if any of the logged messages contain the expected warning
        logged_messages = [call[0][0] for call in logger_func.call_args_list]
        mixed_precision_warning_found = any(
            "Mixed precision training requested but CUDA is not available" in msg
            for msg in logged_messages
        )
        assert mixed_precision_warning_found, f"Expected mixed precision warning not found in: {logged_messages}"


class TestModelManagerUtilities:
    """Test utility methods and information retrieval."""

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_get_model_info(
        self,
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test get_model_info method returns correct information."""
        # Setup mocks
        mock_feature_spec = Mock(num_planes=46)
        mock_features.__getitem__.return_value = mock_feature_spec
        mock_model_factory.return_value = Mock()

        # Create ModelManager
        manager = ModelManager(
            minimal_model_manager_config, mock_args, device, logger_func
        )

        # Get model info
        info = manager.get_model_info()

        # Verify returned information (just check the core fields we care about)
        expected_core_fields = {
            "model_type": minimal_model_manager_config.training.model_type,
            "input_features": minimal_model_manager_config.training.input_features,
            "tower_depth": minimal_model_manager_config.training.tower_depth,
            "tower_width": minimal_model_manager_config.training.tower_width,
            "se_ratio": minimal_model_manager_config.training.se_ratio,
            "obs_shape": (46, 9, 9),
            "num_planes": 46,
            "use_mixed_precision": False,  # CPU device
            "device": "cpu",
        }

        # Check that all expected core fields are present with correct values
        for key, expected_value in expected_core_fields.items():
            assert key in info, f"Missing expected field: {key}"
            assert info[key] == expected_value, f"Field {key}: expected {expected_value}, got {info[key]}"
        
        # Verify torch.compile fields exist (values may vary depending on config)
        assert "torch_compile" in info
        assert "optimization_applied" in info
        assert "performance_benchmarking_enabled" in info

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    @patch("keisei.training.model_manager.safe_compile_model")
    def test_model_creation_and_agent_instantiation(
        self,
        mock_safe_compile_model,
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test model creation through ModelManager."""
        # Setup mocks
        mock_feature_spec = Mock(num_planes=46)
        mock_features.__getitem__.return_value = mock_feature_spec

        # Create a mock model that responds to .to()
        mock_model = Mock()
        mock_model.to.return_value = mock_model  # .to() returns itself
        mock_model_factory.return_value = mock_model
        
        # Mock torch.compile to return the model unchanged for fast test execution
        mock_safe_compile_model.return_value = (mock_model, Mock(success=True, performance_improvement=1.25))

        # Create ModelManager
        manager = ModelManager(
            minimal_model_manager_config, mock_args, device, logger_func
        )

        # Create model
        created_model = manager.create_model()

        # Verify model factory was called with correct parameters
        mock_model_factory.assert_called_once_with(
            model_type=minimal_model_manager_config.training.model_type,
            obs_shape=(46, 9, 9),
            num_actions=minimal_model_manager_config.env.num_actions_total,
            tower_depth=minimal_model_manager_config.training.tower_depth,
            tower_width=minimal_model_manager_config.training.tower_width,
            se_ratio=minimal_model_manager_config.training.se_ratio,
        )

        # Verify model was moved to device
        mock_model.to.assert_called_once_with(device)

        # Verify model is stored and returned
        assert manager.model == mock_model
        assert created_model == mock_model
