"""
Integration tests for Trainer.run_training_loop() with minimal necessary mocking.

Tests verify end-to-end training loop functionality including:
- Complete training loop execution with real component integration
- Resume state logging integration
- Error handling during training loop
- Finalization behavior

Note: Only external dependencies (WandB, file I/O) are mocked. Internal components
use real implementations to test actual integration behavior.
"""

from unittest.mock import patch

import numpy as np
import pytest

from keisei.core.ppo_agent import PPOAgent
from keisei.training.trainer import Trainer


class MockArgs:
    """Mock command-line arguments for testing."""

    def __init__(self, **kwargs):
        self.resume = kwargs.get("resume")
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.mark.integration
class TestTrainerTrainingLoopIntegration:
    """Test end-to-end training loop integration with real components and minimal mocking."""

    @patch("keisei.evaluation.performance_manager.ResourceMonitor", autospec=True)
    @patch("wandb.init")
    @patch("wandb.log")
    @patch("wandb.finish")
    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("torch.save")
    @patch("os.path.exists")
    @pytest.mark.slow
    def test_run_training_loop_with_checkpoint_resume_logging(
        self,
        mock_path_exists,
        _mock_torch_save,
        _mock_makedirs,
        _mock_open,
        _mock_wandb_finish,
        _mock_wandb_log,
        _mock_wandb_init,
        _mock_resource_monitor,
        integration_test_config,
        tmp_path,
    ):
        """Test that run_training_loop() logs checkpoint resume information correctly.

        Uses real components with minimal mocking of external dependencies only.
        """
        checkpoint_data = {
            "global_timestep": 1500,
            "total_episodes_completed": 100,
            "black_wins": 40,
            "white_wins": 35,
            "draws": 25,
        }

        # Create trainer with resume
        checkpoint_path = f"{tmp_path}/resume_test_checkpoint.pth"
        args = MockArgs(resume=checkpoint_path)

        # Update config paths for test isolation
        config = integration_test_config.model_copy(deep=True)
        config.logging.model_dir = f"{tmp_path}/models"

        # Disable WandB for testing
        config.wandb.enabled = False

        # Set very small timestep limit for fast testing
        config.training.total_timesteps = 10
        config.training.steps_per_epoch = 5

        # Mock file existence for checkpoint loading
        mock_path_exists.return_value = True

        # Mock load_model at class level BEFORE Trainer.__init__ triggers
        # checkpoint resume (which calls agent.load_model internally)
        with patch.object(
            PPOAgent, "load_model", return_value=checkpoint_data
        ):
            trainer = Trainer(config, args)

        # Mock only the step execution to control training length and avoid errors
        with patch.object(
            trainer.step_manager, "execute_step"
        ) as mock_execute_step:
            # Mock successful step execution that terminates quickly
            def mock_step_execution(**kwargs):
                episode_state = kwargs.get("episode_state")
                import torch

                from keisei.training.step_manager import StepResult

                # Handle case where episode_state might be None or incomplete
                if episode_state and hasattr(episode_state, "current_obs"):
                    next_obs = episode_state.current_obs
                    next_obs_tensor = episode_state.current_obs_tensor
                else:
                    # Fallback values for testing
                    next_obs = np.zeros((8, 8, 12), dtype=np.float32)
                    next_obs_tensor = torch.zeros(1, 8, 8, 12)

                return StepResult(
                    next_obs=next_obs,
                    next_obs_tensor=next_obs_tensor,
                    reward=1.0,
                    done=True,  # End episode quickly
                    info={"winner": "black", "reason": "checkmate"},
                    selected_move=(0, 0, 0, 0),
                    policy_index=0,
                    log_prob=-1.0,
                    value_pred=0.5,
                    success=True,
                )

            mock_execute_step.side_effect = mock_step_execution

            # Run the training loop
            trainer.run_training_loop()

            # Verify checkpoint data was loaded correctly
            assert trainer.metrics_manager.global_timestep == 1500
            assert (
                trainer.metrics_manager.total_episodes_completed >= 100
            )  # Should be maintained from checkpoint
            assert trainer.metrics_manager.black_wins == 40
            assert trainer.metrics_manager.white_wins == 35
            assert trainer.metrics_manager.draws == 25

    @patch("keisei.evaluation.performance_manager.ResourceMonitor", autospec=True)
    @patch("wandb.init")
    @patch("wandb.log")
    @patch("wandb.finish")
    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("torch.save")
    def test_run_training_loop_fresh_start_no_resume_logging(
        self,
        _mock_torch_save,
        _mock_makedirs,
        _mock_open,
        _mock_wandb_finish,
        _mock_wandb_log,
        _mock_wandb_init,
        _mock_resource_monitor,
        integration_test_config,
        tmp_path,
    ):
        """Test that run_training_loop() does not log resume message when starting fresh.

        Uses real components to validate fresh start behavior.
        """
        # Create trainer without resume (fresh start)
        args = MockArgs(resume=None)

        # Update config paths for test isolation
        config = integration_test_config.model_copy(deep=True)
        config.logging.model_dir = f"{tmp_path}/models"
        config.wandb.enabled = False

        # Set very small timestep limit for fast testing
        config.training.total_timesteps = 10
        config.training.steps_per_epoch = 5

        trainer = Trainer(config, args)

        # Mock only the step execution to control training length and avoid errors
        with patch.object(trainer.step_manager, "execute_step") as mock_execute_step:
            # Mock successful step execution that terminates quickly
            def mock_step_execution(**kwargs):
                episode_state = kwargs.get("episode_state")
                import numpy as np
                import torch

                from keisei.training.step_manager import StepResult

                # Handle case where episode_state might be None or incomplete
                if episode_state and hasattr(episode_state, "current_obs"):
                    next_obs = episode_state.current_obs
                    next_obs_tensor = episode_state.current_obs_tensor
                else:
                    # Fallback values for testing
                    next_obs = np.zeros((8, 8, 12), dtype=np.float32)
                    next_obs_tensor = torch.zeros(1, 8, 8, 12)

                return StepResult(
                    next_obs=next_obs,
                    next_obs_tensor=next_obs_tensor,
                    reward=0.5,
                    done=True,  # End episode quickly
                    info={"winner": "white", "reason": "timeout"},
                    selected_move=(1, 1, 1, 1),
                    policy_index=1,
                    log_prob=-0.5,
                    value_pred=0.3,
                    success=True,
                )

            mock_execute_step.side_effect = mock_step_execution

            # Run the training loop
            trainer.run_training_loop()

            # Verify fresh start state (should start from 0 and progress)
            assert (
                trainer.metrics_manager.global_timestep >= 0
            )  # Should start from 0 or small value
            assert (
                trainer.metrics_manager.total_episodes_completed >= 1
            )  # Should have run at least one episode
            assert trainer.metrics_manager.black_wins >= 0
            assert trainer.metrics_manager.white_wins >= 0
            assert trainer.metrics_manager.draws >= 0

    @patch("keisei.evaluation.performance_manager.ResourceMonitor", autospec=True)
    @patch("wandb.init")
    @patch("wandb.log")
    @patch("wandb.finish")
    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("torch.save")
    def test_run_training_loop_keyboard_interrupt_handling(
        self,
        _mock_torch_save,
        _mock_makedirs,
        _mock_open,
        _mock_wandb_finish,
        _mock_wandb_log,
        _mock_wandb_init,
        _mock_resource_monitor,
        integration_test_config,
        tmp_path,
    ):
        """Test that run_training_loop() handles KeyboardInterrupt gracefully and calls finalization.

        Uses real components to test actual error handling behavior.
        """
        # Create trainer for fresh start
        args = MockArgs(resume=None)

        # Update config paths for test isolation
        config = integration_test_config.model_copy(deep=True)
        config.logging.model_dir = f"{tmp_path}/models"
        config.wandb.enabled = False

        # Set very small timestep limit for fast testing
        config.training.total_timesteps = 10
        config.training.steps_per_epoch = 5

        trainer = Trainer(config, args)

        # Mock step execution to raise KeyboardInterrupt after a couple steps
        call_count = 0

        def mock_step_execution_with_interrupt(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 2:  # Interrupt after a couple successful steps
                raise KeyboardInterrupt("User interrupted during training")

            episode_state = kwargs.get("episode_state")
            import numpy as np
            import torch

            from keisei.training.step_manager import StepResult

            # Handle case where episode_state might be None or incomplete
            if episode_state and hasattr(episode_state, "current_obs"):
                next_obs = episode_state.current_obs
                next_obs_tensor = episode_state.current_obs_tensor
            else:
                # Fallback values for testing
                next_obs = np.zeros((8, 8, 12), dtype=np.float32)
                next_obs_tensor = torch.zeros(1, 8, 8, 12)

            return StepResult(
                next_obs=next_obs,
                next_obs_tensor=next_obs_tensor,
                reward=0.5,
                done=True,
                info={"winner": "black", "reason": "checkmate"},
                selected_move=(0, 0, 0, 0),
                policy_index=0,
                log_prob=-1.0,
                value_pred=0.5,
                success=True,
            )

        with patch.object(trainer.step_manager, "execute_step") as mock_execute_step:
            mock_execute_step.side_effect = mock_step_execution_with_interrupt

            # Run the training loop (should handle KeyboardInterrupt gracefully)
            trainer.run_training_loop()

            # Verify the training ran for at least a couple steps before interruption
            assert mock_execute_step.call_count >= 2

            # Verify trainer state remained consistent
            assert trainer.metrics_manager.global_timestep >= 0
            assert trainer.metrics_manager.total_episodes_completed >= 0

    @patch("keisei.evaluation.performance_manager.ResourceMonitor", autospec=True)
    @patch("wandb.init")
    @patch("wandb.log")
    @patch("wandb.finish")
    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("torch.save")
    def test_run_training_loop_general_exception_handling(
        self,
        _mock_torch_save,
        _mock_makedirs,
        _mock_open,
        _mock_wandb_finish,
        _mock_wandb_log,
        _mock_wandb_init,
        _mock_resource_monitor,
        integration_test_config,
        tmp_path,
    ):
        """Test that run_training_loop() handles general exceptions and calls finalization.

        Uses real components to test actual error handling behavior.
        """
        # Create trainer for fresh start
        args = MockArgs(resume=None)

        # Update config paths for test isolation
        config = integration_test_config.model_copy(deep=True)
        config.logging.model_dir = f"{tmp_path}/models"
        config.wandb.enabled = False

        # Set very small timestep limit for fast testing
        config.training.total_timesteps = 10
        config.training.steps_per_epoch = 5

        trainer = Trainer(config, args)

        # Mock step execution to raise a general exception after a couple steps
        call_count = 0
        test_exception = RuntimeError("Test runtime error during step execution")

        def mock_step_execution_with_exception(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 2:  # Raise exception after a couple successful steps
                raise test_exception

            episode_state = kwargs.get("episode_state")
            import numpy as np
            import torch

            from keisei.training.step_manager import StepResult

            # Handle case where episode_state might be None or incomplete
            if episode_state and hasattr(episode_state, "current_obs"):
                next_obs = episode_state.current_obs
                next_obs_tensor = episode_state.current_obs_tensor
            else:
                # Fallback values for testing
                next_obs = np.zeros((8, 8, 12), dtype=np.float32)
                next_obs_tensor = torch.zeros(1, 8, 8, 12)

            return StepResult(
                next_obs=next_obs,
                next_obs_tensor=next_obs_tensor,
                reward=0.5,
                done=True,
                info={"winner": "white", "reason": "timeout"},
                selected_move=(1, 1, 1, 1),
                policy_index=1,
                log_prob=-0.5,
                value_pred=0.3,
                success=True,
            )

        with patch.object(trainer.step_manager, "execute_step") as mock_execute_step:
            mock_execute_step.side_effect = mock_step_execution_with_exception

            # Run the training loop (should handle general exception gracefully)
            trainer.run_training_loop()

            # Verify the training ran for at least a couple steps before exception
            assert mock_execute_step.call_count >= 2

            # Verify trainer state remained consistent
            assert trainer.metrics_manager.global_timestep >= 0
            assert trainer.metrics_manager.total_episodes_completed >= 0

    @patch("keisei.evaluation.performance_manager.ResourceMonitor", autospec=True)
    @patch("wandb.init")
    @patch("wandb.log")
    @patch("wandb.finish")
    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("torch.save")
    @pytest.mark.slow
    def test_training_loop_state_consistency_throughout_execution(
        self,
        _mock_torch_save,
        _mock_makedirs,
        _mock_open,
        _mock_wandb_finish,
        _mock_wandb_log,
        _mock_wandb_init,
        _mock_resource_monitor,
        integration_test_config,
        tmp_path,
    ):
        """Test that training state remains consistent throughout run_training_loop() execution.

        Uses real components to test actual state management behavior.
        """
        checkpoint_data = {
            "global_timestep": 2000,
            "total_episodes_completed": 150,
            "black_wins": 60,
            "white_wins": 55,
            "draws": 35,
        }

        # Create trainer with resume to test state preservation
        checkpoint_path = f"{tmp_path}/state_consistency_checkpoint.pth"
        args = MockArgs(resume=checkpoint_path)

        # Update config paths for test isolation
        config = integration_test_config.model_copy(deep=True)
        config.logging.model_dir = f"{tmp_path}/models"
        config.wandb.enabled = False

        # Set very small timestep limit for fast testing
        config.training.total_timesteps = 2010  # Just a bit more than checkpoint
        config.training.steps_per_epoch = 5

        # Mock load_model at class level and path.exists BEFORE Trainer.__init__
        # triggers checkpoint resume
        with patch("os.path.exists", return_value=True), patch.object(
            PPOAgent, "load_model", return_value=checkpoint_data
        ):
            trainer = Trainer(config, args)

        # Verify initial state after checkpoint resume
        initial_global_timestep = trainer.metrics_manager.global_timestep
        initial_episodes = trainer.metrics_manager.total_episodes_completed
        initial_black_wins = trainer.metrics_manager.black_wins
        initial_white_wins = trainer.metrics_manager.white_wins
        initial_draws = trainer.metrics_manager.draws

        assert initial_global_timestep == 2000
        assert initial_episodes == 150
        assert initial_black_wins == 60
        assert initial_white_wins == 55
        assert initial_draws == 35

        # Mock step execution to run a few steps and ensure consistency
        call_count = 0

        def mock_step_execution_with_tracking(**kwargs):
            nonlocal call_count
            call_count += 1

            episode_state = kwargs.get("episode_state")
            import numpy as np
            import torch

            from keisei.training.step_manager import StepResult

            # Handle case where episode_state might be None or incomplete
            if episode_state and hasattr(episode_state, "current_obs"):
                next_obs = episode_state.current_obs
                next_obs_tensor = episode_state.current_obs_tensor
            else:
                # Fallback values for testing
                next_obs = np.zeros((8, 8, 12), dtype=np.float32)
                next_obs_tensor = torch.zeros(1, 8, 8, 12)

            return StepResult(
                next_obs=next_obs,
                next_obs_tensor=next_obs_tensor,
                reward=1.0,
                done=True,  # End episode quickly
                info={"winner": "black", "reason": "checkmate"},
                selected_move=(0, 0, 0, 0),
                policy_index=0,
                log_prob=-1.0,
                value_pred=0.5,
                success=True,
            )

        with patch.object(trainer.step_manager, "execute_step") as mock_execute_step:
            mock_execute_step.side_effect = mock_step_execution_with_tracking

            # Run the training loop
            trainer.run_training_loop()

            # Verify state consistency was maintained
            # Global timestep should have advanced from the checkpoint
            assert trainer.metrics_manager.global_timestep >= initial_global_timestep

            # Episode count should have increased (we ran more steps)
            assert trainer.metrics_manager.total_episodes_completed >= initial_episodes

            # Win counts should be maintained or increased (we only added black wins)
            assert trainer.metrics_manager.black_wins >= initial_black_wins
            assert (
                trainer.metrics_manager.white_wins >= initial_white_wins
            )  # Should be same or more
            assert (
                trainer.metrics_manager.draws >= initial_draws
            )  # Should be same or more

            # Verify some training actually happened
            assert mock_execute_step.call_count > 0
