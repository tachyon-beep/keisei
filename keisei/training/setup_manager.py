"""
setup_manager.py: Handles complex initialization and setup logic for the Trainer class.
"""

import sys
from datetime import datetime
from typing import Any, Optional, Tuple

import torch

from keisei.config_schema import AppConfig
from keisei.core.actor_critic_protocol import ActorCriticProtocol
from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.ppo_agent import PPOAgent
from keisei.utils import TrainingLogger
from keisei.utils.unified_logger import log_error_to_stderr

from .step_manager import StepManager


class SetupManager:
    """
    Manages the complex setup and initialization logic for training components.
    Extracts detailed setup methods from the main Trainer class.
    """

    def __init__(self, config: AppConfig, device: torch.device):
        """
        Initialize the SetupManager.

        Args:
            config: Application configuration
            device: PyTorch device for training
        """
        self.config = config
        self.device = device

    def setup_game_components(self, env_manager):
        """
        Initialize game environment and policy mapper using EnvManager.

        Args:
            env_manager: Environment manager instance

        Returns:
            Tuple of (game, policy_output_mapper, action_space_size, obs_space_shape)
        """
        try:
            # Call EnvManager's setup_environment to get game and mapper
            game, policy_output_mapper = env_manager.setup_environment()

            # Retrieve environment info
            action_space_size = env_manager.action_space_size
            obs_space_shape = env_manager.obs_space_shape

            if game is None or policy_output_mapper is None:
                raise RuntimeError(
                    "EnvManager.setup_environment() failed to return valid game or policy_output_mapper."
                )

            return game, policy_output_mapper, action_space_size, obs_space_shape

        except (RuntimeError, ValueError, OSError) as e:
            print(
                f"Error initializing game components: {e}. Aborting.",
                file=sys.stderr,
            )
            raise RuntimeError(f"Failed to initialize game components: {e}") from e

    def setup_training_components(self, model_manager):
        """
        Initialize PPO agent and experience buffer.

        Args:
            model_manager: Model manager instance

        Returns:
            Tuple of (model, agent, experience_buffer)
        """
        # Create model using ModelManager
        model = model_manager.create_model()

        if model is None:
            raise RuntimeError(
                "Model was not created successfully before agent initialization."
            )

        # Initialize PPOAgent with the model (dependency injection)
        agent = PPOAgent(
            model=model,
            config=self.config,
            device=self.device,
            scaler=model_manager.scaler,
            use_mixed_precision=model_manager.use_mixed_precision,
        )

        experience_buffer = ExperienceBuffer(
            buffer_size=self.config.training.steps_per_epoch,
            gamma=self.config.training.gamma,
            lambda_gae=self.config.training.lambda_gae,
            device=self.config.env.device,
        )

        return model, agent, experience_buffer

    def setup_step_manager(self, game, agent, policy_output_mapper, experience_buffer):
        """
        Initialize StepManager for step execution and episode management.

        Args:
            game: Game environment instance
            agent: PPO agent instance
            policy_output_mapper: Policy output mapper
            experience_buffer: Experience buffer instance

        Returns:
            Configured StepManager instance
        """
        step_manager = StepManager(
            config=self.config,
            game=game,
            agent=agent,
            policy_mapper=policy_output_mapper,
            experience_buffer=experience_buffer,
        )
        return step_manager

    def handle_checkpoint_resume(
        self,
        model_manager,
        agent,
        model_dir,
        resume_path_override,
        metrics_manager,
        logger,
    ):
        """
        Handle resuming from checkpoint using ModelManager.

        Args:
            model_manager: Model manager instance
            agent: PPO agent instance
            model_dir: Model directory path
            resume_path_override: Optional resume path override
            metrics_manager: Metrics manager instance
            logger: Logger instance

        Returns:
            True if resumed from checkpoint, False otherwise
        """
        if not agent:
            logger.log(
                "[ERROR] Agent not initialized before handling checkpoint resume. This should not happen."
            )
            raise RuntimeError("Agent not initialized before _handle_checkpoint_resume")

        model_manager.handle_checkpoint_resume(
            agent=agent,
            model_dir=model_dir,
            resume_path_override=resume_path_override,
        )

        resumed_from_checkpoint = model_manager.resumed_from_checkpoint

        # Restore training state from checkpoint data
        if model_manager.checkpoint_data:
            checkpoint_data = model_manager.checkpoint_data
            metrics_manager.restore_from_checkpoint(checkpoint_data)

        return resumed_from_checkpoint

    def log_event(self, message: str, log_file_path: str):
        """Log important events to the main training log file."""
        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
        except (OSError, IOError) as e:
            log_error_to_stderr("SetupManager", "Failed to log event", e)

    def log_run_info(
        self, session_manager, model_manager, agent, metrics_manager, log_both
    ):
        """Log run information at the start of training."""
        # Delegate session info logging to SessionManager
        agent_name = "N/A"
        agent_type_name = "N/A"
        if agent:
            agent_name = getattr(agent, "name", "N/A")
            agent_type_name = type(agent).__name__

        agent_info = {"type": agent_type_name, "name": agent_name}

        def log_wrapper(msg):
            log_both(msg)

        session_manager.log_session_info(
            logger_func=log_wrapper,
            agent_info=agent_info,
            resumed_from_checkpoint=model_manager.resumed_from_checkpoint,
            global_timestep=metrics_manager.global_timestep,
            total_episodes_completed=metrics_manager.total_episodes_completed,
        )

        # Log model structure using ModelManager
        model_info = model_manager.get_model_info()
        log_both(f"Model Structure:\n{model_info}", also_to_wandb=False)

        # Log to main training log file
        self.log_event(f"Model Structure:\n{model_info}", session_manager.log_file_path)
