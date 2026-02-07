"""
Unit tests for ModelManager checkpoint handling and loading functionality.
"""

import os
import sys
import tempfile
from unittest.mock import Mock, patch

import pytest
import torch  # Added torch import

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
from keisei.training.model_manager import ModelManager  # Updated import path


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
            num_actions_total=13527,
            input_channels=46,
            seed=42,
            max_moves_per_game=500,
        ),
        training=TrainingConfig(
            total_timesteps=500_000,
            steps_per_epoch=2048,
            ppo_epochs=10,
            minibatch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=True,
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
            enable_periodic_evaluation=True,
            strategy="single_opponent",
            num_games=20,
            max_concurrent_games=4,
            timeout_per_game=None,
            opponent_type="random",
            evaluation_interval_timesteps=50000,
            max_moves_per_game=500,
            randomize_positions=True,
            random_seed=None,
            save_games=True,
            save_path=None,
            log_file_path_eval="eval_log.txt",
            log_level="INFO",
            wandb_log_eval=False,
            update_elo=True,
            elo_registry_path="elo_ratings.json",
            agent_id=None,
            opponent_id=None,
            previous_model_pool_size=5,
            enable_in_memory_evaluation=True,
            model_weight_cache_size=5,
            enable_parallel_execution=True,
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
        display=DisplayConfig(
            enable_board_display=True,
            enable_trend_visualization=True,
            enable_elo_ratings=True,
            enable_enhanced_layout=True,
            display_moves=False,
            turn_tick=0.5,
            board_unicode_pieces=True,
            board_cell_width=5,
            board_cell_height=3,
            board_highlight_last_move=True,
            sparkline_width=15,
            trend_history_length=100,
            elo_initial_rating=1500.0,
            elo_k_factor=32.0,
            dashboard_height_ratio=2,
            progress_bar_height=4,
            show_text_moves=True,
            move_list_length=10,
            moves_latest_top=True,
            moves_flash_ms=500,
            show_moves_trend=True,
            show_completion_rate=True,
            show_enhanced_win_rates=True,
            show_turns_trend=True,
            metrics_window_size=100,
            trend_smoothing_factor=0.1,
            metrics_panel_height=6,
            enable_trendlines=True,
            log_layer_keyword_filters=["stem", "policy_head", "value_head"],
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
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


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
    )


@pytest.fixture
def mock_feature_spec():
    """Create a consistent mock feature spec."""
    mock_spec = Mock()
    mock_spec.num_planes = 46
    return mock_spec


class TestModelManagerCheckpointHandling:
    """Test checkpoint loading and resuming functionality."""

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    @patch("keisei.training.model_manager.utils.find_latest_checkpoint")
    def test_handle_checkpoint_resume_latest_found(
        self,
        mock_find_checkpoint,
        mock_model_factory,
        mock_feature_specs,
        minimal_model_manager_config,
        device,
        logger_func,
        temp_dir,
        mock_feature_spec,
    ):
        """Test resuming from latest checkpoint when found."""
        # Setup mocks
        mock_feature_specs["core46"] = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        checkpoint_path = os.path.join(temp_dir, "checkpoint.pth")
        mock_find_checkpoint.return_value = checkpoint_path

        # Create args with resume="latest"
        args = MockArgs(resume="latest")

        # Create ModelManager
        manager = ModelManager(minimal_model_manager_config, args, device, logger_func)

        # Create mock agent with realistic load_model return value
        mock_agent = Mock()
        mock_agent.load_model.return_value = {
            "global_timestep": 100,
            "total_episodes_completed": 10,
            "black_wins": 5,
            "white_wins": 3,
            "draws": 2,
        }

        # Test checkpoint resume
        result = manager.handle_checkpoint_resume(mock_agent, temp_dir)

        # Verify checkpoint loaded
        assert result is True
        assert manager.resumed_from_checkpoint == checkpoint_path
        mock_agent.load_model.assert_called_once_with(checkpoint_path)
        logger_func.assert_any_call(
            f"Resumed from latest checkpoint: {checkpoint_path}"
        )

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    @patch("keisei.training.model_manager.utils.find_latest_checkpoint")
    def test_handle_checkpoint_resume_not_found(
        self,
        mock_find_checkpoint,
        mock_model_factory,
        mock_feature_specs,
        minimal_model_manager_config,
        device,
        logger_func,
        temp_dir,
        mock_feature_spec,
    ):
        """Test resuming when no checkpoint found."""
        # Setup mocks
        mock_feature_specs["core46"] = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_find_checkpoint.return_value = None

        # Create args with resume="latest"
        args = MockArgs(resume="latest")

        # Create ModelManager
        manager = ModelManager(minimal_model_manager_config, args, device, logger_func)

        # Create mock agent
        mock_agent = Mock()

        # Test checkpoint resume
        result = manager.handle_checkpoint_resume(mock_agent, temp_dir)

        # Verify no checkpoint loaded
        assert result is False
        assert manager.resumed_from_checkpoint is None
        mock_agent.load_model.assert_not_called()

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    @patch("os.path.exists")  # Add patch for os.path.exists
    def test_handle_checkpoint_resume_explicit_path(
        self,
        mock_os_path_exists,  # Add mock for os.path.exists
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        device,
        logger_func,
        temp_dir,
        mock_feature_spec,
    ):
        """Test resuming from explicit checkpoint path."""
        # Setup mocks
        mock_features.FEATURE_SPECS = {"core46": mock_feature_spec}

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        checkpoint_path = "/path/to/specific/checkpoint.pth"
        mock_os_path_exists.return_value = True

        # Create args with explicit resume path
        args = MockArgs(resume=checkpoint_path)

        # Create ModelManager
        manager = ModelManager(minimal_model_manager_config, args, device, logger_func)

        # Create mock agent with realistic load_model return value
        mock_agent = Mock()
        mock_agent.load_model.return_value = {
            "global_timestep": 100,
            "total_episodes_completed": 10,
            "black_wins": 5,
            "white_wins": 3,
            "draws": 2,
        }

        # Test checkpoint resume
        result = manager.handle_checkpoint_resume(mock_agent, temp_dir)

        # Verify checkpoint loaded
        assert result is True
        assert manager.resumed_from_checkpoint == checkpoint_path
        mock_agent.load_model.assert_called_once_with(checkpoint_path)
        logger_func.assert_any_call(
            f"Resumed from specified checkpoint: {checkpoint_path}"
        )


class TestModelManagerEnhancedCheckpointHandling:
    """Enhanced tests for checkpoint loading scenarios and edge cases."""

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    @patch("keisei.training.model_manager.utils.find_latest_checkpoint")
    def test_load_checkpoint_multiple_available(
        self,
        mock_find_checkpoint,
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        device,
        logger_func,
    ):
        """Test checkpoint loading when multiple checkpoints exist."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Mock find_latest_checkpoint to return the latest checkpoint
        latest_checkpoint_path = os.path.join(
            minimal_model_manager_config.logging.model_dir, "checkpoint_2000.pth"
        )
        mock_find_checkpoint.return_value = latest_checkpoint_path

        # Mock checkpoint data
        checkpoint_data = {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "global_timestep": 2000,
            "total_episodes_completed": 50,
            "black_wins": 25,
            "white_wins": 20,
            "draws": 5,
        }

        # Create args with resume="latest"
        args_with_resume = MockArgs(resume="latest")
        manager = ModelManager(
            minimal_model_manager_config, args_with_resume, device, logger_func
        )
        # Mock create_model to avoid torch.compile optimization in tests
        with patch.object(manager, 'create_model') as mock_create_model:
            mock_create_model.return_value = mock_model
            manager.model = mock_model

            # Mock an agent for testing
            mock_agent = Mock()
            mock_agent.load_model.return_value = checkpoint_data

            # Test loading latest checkpoint
            result = manager.handle_checkpoint_resume(
                mock_agent, minimal_model_manager_config.logging.model_dir
            )

            assert result is True
            assert manager.checkpoint_data is not None
            assert manager.checkpoint_data["global_timestep"] == 2000
            mock_agent.load_model.assert_called_with(latest_checkpoint_path)

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    @patch("os.path.exists")
    def test_load_checkpoint_specific_not_found(
        self,
        mock_exists,
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test loading a specific checkpoint that doesn't exist."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Mock specific checkpoint doesn't exist
        mock_exists.return_value = False

        manager = ModelManager(
            minimal_model_manager_config, mock_args, device, logger_func
        )
        
        # Mock create_model to avoid torch.compile optimization in tests
        with patch.object(manager, 'create_model') as mock_create_model:
            mock_create_model.return_value = mock_model
            manager.model = mock_model

            mock_agent = Mock()

            # Test loading specific non-existent checkpoint - set args to use specific path
            args_with_resume = MockArgs(resume="checkpoint_9999.pth")
            manager_with_resume = ModelManager(
                minimal_model_manager_config, args_with_resume, device, logger_func
            )
            
            # Mock create_model for the second manager too
            with patch.object(manager_with_resume, 'create_model') as mock_create_model2:
                mock_create_model2.return_value = mock_model
                manager_with_resume.model = mock_model

                result = manager_with_resume.handle_checkpoint_resume(
                    mock_agent, "/some/model/dir"
                )

                assert result is False
                logger_func.assert_any_call(
                    "Specified resume checkpoint not found: checkpoint_9999.pth"
                )

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    @patch("os.path.exists")
    @patch("torch.load")
    def test_load_checkpoint_corrupted_data(
        self,
        mock_torch_load,
        mock_exists,
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test handling of corrupted checkpoint data."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_exists.return_value = True
        # Mock corrupted checkpoint (missing required keys)
        mock_torch_load.return_value = {"incomplete": "data"}

        manager = ModelManager(
            minimal_model_manager_config, mock_args, device, logger_func
        )
        
        # Mock create_model to avoid torch.compile optimization in tests
        with patch.object(manager, 'create_model') as mock_create_model:
            mock_create_model.return_value = mock_model
            manager.model = mock_model

            mock_agent = Mock()
            mock_agent.load_model.return_value = {"incomplete": "data"}

            # Test loading corrupted checkpoint
            args_with_resume = MockArgs(resume="latest")
            manager_with_resume = ModelManager(
                minimal_model_manager_config, args_with_resume, device, logger_func
            )
            
            # Mock create_model for the second manager too
            with patch.object(manager_with_resume, 'create_model') as mock_create_model2:
                mock_create_model2.return_value = mock_model
                manager_with_resume.model = mock_model

                # Mock the utils.find_latest_checkpoint to return a valid path
                with patch(
                    "keisei.training.model_manager.utils.find_latest_checkpoint",
                    return_value="/path/to/corrupt.pth",
                ):
                    result = manager_with_resume.handle_checkpoint_resume(
                        mock_agent, "/some/model/dir"
                    )

                # Should handle gracefully - either succeed or fail, but not crash
                assert isinstance(result, bool)  # Should return a boolean

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_save_checkpoint_directory_creation(
        self,
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test that save_checkpoint creates directories if they don't exist."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Use a non-existent subdirectory
        nonexistent_model_dir = os.path.join(temp_dir, "models", "subdir")
        minimal_model_manager_config.logging.model_dir = nonexistent_model_dir

        manager = ModelManager(
            minimal_model_manager_config, mock_args, device, logger_func
        )
        
        # Mock the create_model method to avoid torch.compile optimization in tests
        with patch.object(manager, 'create_model') as mock_create_model:
            mock_create_model.return_value = mock_model
            manager.model = mock_model  # Set the model directly

            mock_agent = Mock()
            mock_agent.model = manager.model
            mock_agent.optimizer = Mock()
            # Add proper mock for save_model method to prevent infinite loop
            mock_agent.save_model = Mock(return_value=True)

            # Test checkpoint saving - should create directory
            stats = {
                "black_wins": 10,
                "white_wins": 8,
                "draws": 7,
            }
            
            # Mock WandB to prevent network calls during test
            mock_wandb = Mock()
            mock_wandb.run = None  # Disable WandB during test
            with patch.dict(sys.modules, {"wandb": mock_wandb}):
                result = manager.save_checkpoint(
                    agent=mock_agent,
                    model_dir=nonexistent_model_dir,
                    timestep=1000,
                    episode_count=25,
                    stats=stats,
                    run_name="test_run",
                    is_wandb_active=False,
                )

            # Verify checkpoint save was successful
            success, checkpoint_path = result
            assert success is True
            assert checkpoint_path is not None
            
            # Verify directory was created
            assert os.path.exists(nonexistent_model_dir)

            # Verify save was attempted via agent.save_model
            mock_agent.save_model.assert_called_once_with(
                os.path.join(nonexistent_model_dir, "checkpoint_ts1000.pth"),
                1000,
                25,
                stats_to_save=stats,
            )
