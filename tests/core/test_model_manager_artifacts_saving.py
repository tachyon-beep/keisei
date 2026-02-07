"""
Unit tests for ModelManager artifact creation and saving functionality.
"""

import os
import sys
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


class TestModelManagerArtifacts:
    """Test WandB artifact creation functionality."""

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_create_model_artifact_success(
        self,
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test successful model artifact creation."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.FEATURE_SPECS = {"core46": mock_feature_spec}

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Create a test model file
        model_path = os.path.join(temp_dir, "test_model.pth")
        with open(model_path, "w", encoding="utf-8") as f:
            f.write("test model content")

        # Setup WandB mock via sys.modules (lazy import in create_model_artifact)
        mock_wandb = Mock()
        mock_artifact = Mock()
        mock_wandb.Artifact.return_value = mock_artifact
        mock_wandb.run = Mock()

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            # Create ModelManager
            manager = ModelManager(
                minimal_model_manager_config, mock_args, device, logger_func
            )

            # Test artifact creation
            result = manager.create_model_artifact(
                model_path=model_path,
                artifact_name="test-model",
                run_name="test_run",
                is_wandb_active=True,
                description="Test model",
                metadata={"test": "value"},
                aliases=["latest"],
            )

        # Verify artifact created
        assert result is True
        mock_wandb.Artifact.assert_called_once()
        mock_artifact.add_file.assert_called_once_with(model_path)
        mock_wandb.log_artifact.assert_called_once_with(
            mock_artifact, aliases=["latest"]
        )

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_create_model_artifact_wandb_inactive(
        self,
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test artifact creation when WandB is inactive."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.FEATURE_SPECS = {"core46": mock_feature_spec}

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Create ModelManager
        manager = ModelManager(
            minimal_model_manager_config, mock_args, device, logger_func
        )

        # Test artifact creation with WandB inactive
        result = manager.create_model_artifact(
            model_path="/path/to/model.pth",
            artifact_name="test-model",
            run_name="test_run",
            is_wandb_active=False,
        )

        # Verify artifact not created
        assert result is False

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_create_model_artifact_file_missing(
        self,
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test artifact creation when model file is missing."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Setup WandB mock via sys.modules (lazy import)
        mock_wandb = Mock()
        mock_wandb.run = Mock()

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            # Create ModelManager
            manager = ModelManager(
                minimal_model_manager_config, mock_args, device, logger_func
            )

            # Test artifact creation with missing file
            result = manager.create_model_artifact(
                model_path="/nonexistent/model.pth",
                artifact_name="test-model",
                run_name="test_run",
                is_wandb_active=True,
            )

        # Verify artifact not created
        assert result is False
        logger_func.assert_any_call(
            "Warning: Model file /nonexistent/model.pth does not exist, skipping artifact creation."
        )


class TestModelManagerSaving:
    """Test model and checkpoint saving functionality."""

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_save_final_model_success(
        self,
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test successful final model saving."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.FEATURE_SPECS = {"core46": mock_feature_spec}

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Create ModelManager
        manager = ModelManager(
            minimal_model_manager_config, mock_args, device, logger_func
        )

        # Create mock agent
        mock_agent = Mock()

        game_stats = {"black_wins": 10, "white_wins": 5, "draws": 2}

        # Test final model saving
        with patch.object(
            manager, "create_model_artifact", return_value=True
        ) as mock_artifact:
            success, model_path = manager.save_final_model(
                agent=mock_agent,
                model_dir=temp_dir,
                global_timestep=1000,
                total_episodes_completed=17,
                game_stats=game_stats,
                run_name="test_run",
                is_wandb_active=True,
            )

        # Verify model saved
        assert success is True
        assert model_path == os.path.join(temp_dir, "final_model.pth")
        mock_agent.save_model.assert_called_once_with(model_path, 1000, 17)
        mock_artifact.assert_called_once()

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_save_final_checkpoint_success(
        self,
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test successful final checkpoint saving."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.FEATURE_SPECS = {"core46": mock_feature_spec}

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Create ModelManager
        manager = ModelManager(
            minimal_model_manager_config, mock_args, device, logger_func
        )

        # Create mock agent
        mock_agent = Mock()

        game_stats = {"black_wins": 10, "white_wins": 5, "draws": 2}

        # Test final checkpoint saving
        with patch.object(
            manager, "create_model_artifact", return_value=True
        ) as mock_artifact:
            success, checkpoint_path = manager.save_final_checkpoint(
                agent=mock_agent,
                model_dir=temp_dir,
                global_timestep=1000,
                total_episodes_completed=17,
                game_stats=game_stats,
                run_name="test_run",
                is_wandb_active=True,
            )

        # Verify checkpoint saved
        assert success is True
        expected_path = os.path.join(temp_dir, "checkpoint_ts1000.pth")
        assert checkpoint_path == expected_path
        mock_agent.save_model.assert_called_once_with(
            expected_path, 1000, 17, stats_to_save=game_stats
        )
        mock_artifact.assert_called_once()

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_save_final_checkpoint_zero_timestep(
        self,
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test final checkpoint saving with zero timestep."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.FEATURE_SPECS = {"core46": mock_feature_spec}

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Create ModelManager
        manager = ModelManager(
            minimal_model_manager_config, mock_args, device, logger_func
        )

        # Create mock agent
        mock_agent = Mock()

        game_stats = {"black_wins": 0, "white_wins": 0, "draws": 0}

        # Test final checkpoint saving with zero timestep
        success, checkpoint_path = manager.save_final_checkpoint(
            agent=mock_agent,
            model_dir=temp_dir,
            global_timestep=0,
            total_episodes_completed=0,
            game_stats=game_stats,
            run_name="test_run",
            is_wandb_active=True,
        )

        # Verify checkpoint not saved
        assert success is False
        assert checkpoint_path is None
        mock_agent.save_model.assert_not_called()

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


class TestModelManagerWandBArtifactEnhancements:
    """Enhanced tests for W&B artifact creation with edge cases."""

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    @patch("wandb.Artifact")
    @patch("wandb.log_artifact")
    @patch("wandb.run")
    def test_create_model_artifact_with_metadata(
        self,
        mock_wandb_run,
        mock_log_artifact,
        mock_artifact_class,
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test model artifact creation with comprehensive metadata."""
        # Setup mocks
        mock_wandb_run.return_value = True  # Mock that wandb.run is active
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact

        # Create test model file
        model_path = os.path.join(temp_dir, "test_model.pth")
        with open(model_path, "w", encoding="utf-8") as f:
            f.write("dummy model data")

        manager = ModelManager(
            minimal_model_manager_config, mock_args, device, logger_func
        )

        # Test artifact creation with metadata
        result = manager.create_model_artifact(
            model_path=model_path,
            artifact_name="enhanced-test-model",
            run_name="test_run_with_metadata",
            is_wandb_active=True,
            metadata={"epochs": 100, "accuracy": 0.95},
        )

        # Verify artifact creation was attempted (mocked W&B environment)
        assert result is True
        mock_artifact_class.assert_called_once()
        mock_log_artifact.assert_called_once()
        mock_artifact_class.assert_called_once()
        mock_artifact.add_file.assert_called_once_with(model_path)
        mock_log_artifact.assert_called_once_with(mock_artifact, aliases=None)

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    @patch("wandb.Artifact")
    @patch("wandb.log_artifact", side_effect=RuntimeError("W&B API Error"))
    @patch("wandb.run")
    def test_create_model_artifact_wandb_failure_handling(
        self,
        mock_wandb_run,
        mock_log_artifact,  # Keep but mark as used for side_effect
        mock_artifact_class,
        mock_model_factory,
        mock_features,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test robust handling of W&B API failures during artifact creation."""
        # Setup mocks
        mock_wandb_run.return_value = True  # Mock that wandb.run is active
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact

        # Create test model file
        model_path = os.path.join(temp_dir, "test_model.pth")
        with open(model_path, "w", encoding="utf-8") as f:
            f.write("dummy model data")

        manager = ModelManager(
            minimal_model_manager_config, mock_args, device, logger_func
        )

        # Verify the mock_log_artifact is configured to raise an error
        assert mock_log_artifact.side_effect is not None

        # Test artifact creation with W&B failure
        result = manager.create_model_artifact(
            model_path=model_path,
            artifact_name="failing-test-model",
            run_name="test_run_fail",
            is_wandb_active=True,
        )

        # Should handle failure gracefully
        assert result is False
        logger_func.assert_any_call(
            f"Error creating W&B artifact for {model_path}: W&B API Error"
        )

    def test_create_model_artifact_wandb_inactive(
        self,
        minimal_model_manager_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test that artifact creation is skipped when W&B is inactive."""
        with (
            patch("keisei.shogi.features.FEATURE_SPECS") as mock_features,
            patch("keisei.training.models.model_factory") as mock_model_factory,
        ):

            # Setup mocks
            mock_feature_spec = Mock()
            mock_feature_spec.num_planes = 46
            mock_features.__getitem__.return_value = mock_feature_spec

            mock_model = Mock()
            mock_model.to.return_value = mock_model
            mock_model_factory.return_value = mock_model

            # Create test model file
            model_path = os.path.join(temp_dir, "test_model.pth")
            with open(model_path, "w", encoding="utf-8") as f:
                f.write("dummy model data")

            manager = ModelManager(
                minimal_model_manager_config, mock_args, device, logger_func
            )

            # Test with W&B inactive
            result = manager.create_model_artifact(
                model_path=model_path,
                artifact_name="inactive-wandb-test",
                run_name="test_run",
                is_wandb_active=False,  # W&B not active
            )

            # Should skip artifact creation
            assert result is False
            # The method returns False immediately when W&B is inactive,
            # without logging a specific message
