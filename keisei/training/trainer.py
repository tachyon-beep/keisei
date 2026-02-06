"""
trainer.py: Contains the Trainer class for managing the Shogi RL training loop (refactored).
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

import torch

import wandb
from keisei.config_schema import AppConfig
from keisei.core.actor_critic_protocol import ActorCriticProtocol
from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.ppo_agent import PPOAgent
from keisei.evaluation.enhanced_manager import EnhancedEvaluationManager
from keisei.utils import TrainingLogger

from .callback_manager import CallbackManager
from .display_manager import DisplayManager
from .env_manager import EnvManager
from .metrics_manager import MetricsManager
from .model_manager import ModelManager
from .session_manager import SessionManager
from .setup_manager import SetupManager
from .step_manager import EpisodeState, StepManager
from .training_loop_manager import TrainingLoopManager


class Trainer:
    """
    Manages the training process for the PPO Shogi agent.
    This class orchestrates setup, training loop, evaluation, and logging.
    """

    def __init__(self, config: AppConfig, args: Any):
        """
        Initializes the Trainer with configuration and command-line arguments.

        Args:
            config: An AppConfig Pydantic object containing all configuration.
            args: Parsed command-line arguments.
        """
        self.config = config
        self.args = args

        # Core instance attributes
        self.model: Optional[ActorCriticProtocol] = None
        self.agent: Optional[PPOAgent] = None
        self.experience_buffer: Optional[ExperienceBuffer] = None
        self.step_manager: Optional[StepManager] = None
        self.game = None
        self.policy_output_mapper = None
        self.action_space_size = 0
        self.obs_space_shape = ()
        self.log_both: Optional[Callable] = None
        self.resumed_from_checkpoint = False

        # Initialize device
        self.device = torch.device(config.env.device)

        # Initialize session manager and setup session infrastructure
        self.session_manager = SessionManager(config, args)
        self.session_manager.setup_directories()
        self.session_manager.setup_wandb()
        self.session_manager.save_effective_config()
        self.session_manager.setup_seeding()

        # Access session properties
        self.run_name = self.session_manager.run_name
        self.run_artifact_dir = self.session_manager.run_artifact_dir
        self.model_dir = self.session_manager.model_dir
        self.log_file_path = self.session_manager.log_file_path
        self.eval_log_file_path = self.session_manager.eval_log_file_path
        self.is_train_wandb_active = self.session_manager.is_wandb_active

        # Initialize managers
        self.display_manager = DisplayManager(config, self.log_file_path)
        self.rich_console = self.display_manager.get_console()
        self.rich_log_messages = self.display_manager.get_log_messages()
        self.logger = TrainingLogger(
            self.log_file_path,
            self.rich_console,
            self.rich_log_messages,
            also_stdout=False,
        )

        # Initialize WebUI manager if enabled
        self.webui_manager = None
        self.webui_http_server = None
        if config.webui.enabled:
            try:
                from keisei.webui.webui_manager import WebUIManager
                from keisei.webui.web_server import WebUIHTTPServer
                
                self.webui_manager = WebUIManager(config.webui)
                self.webui_http_server = WebUIHTTPServer(
                    host=config.webui.host,
                    port=config.webui.port + 1  # HTTP server on port+1
                )
                
                # Start WebUI servers
                if self.webui_manager.start():
                    self.logger.log("WebUI WebSocket server started successfully")
                else:
                    self.logger.log("Failed to start WebUI WebSocket server")
                    self.webui_manager = None
                    
                if self.webui_http_server.start():
                    self.logger.log(f"WebUI HTTP server available at http://{config.webui.host}:{config.webui.port + 1}")
                else:
                    self.logger.log("Failed to start WebUI HTTP server")
                    self.webui_http_server = None
                    
            except ImportError as e:
                self.logger.log(f"WebUI not available: {e}")
                self.webui_manager = None
                self.webui_http_server = None
        self.model_manager = ModelManager(config, args, self.device, self.logger.log)
        self.env_manager = EnvManager(config, self.logger.log)
        self.metrics_manager = MetricsManager(
            history_size=config.display.trend_history_length,
            elo_initial_rating=config.display.elo_initial_rating,
            elo_k_factor=config.display.elo_k_factor,
        )
        # Create evaluation config directly from central config
        from keisei.evaluation.core import create_evaluation_config

        eval_config = create_evaluation_config(
            strategy=config.evaluation.strategy,
            num_games=config.evaluation.num_games,
            max_concurrent_games=config.evaluation.max_concurrent_games,
            timeout_per_game=config.evaluation.timeout_per_game,
            randomize_positions=config.evaluation.randomize_positions,
            random_seed=config.evaluation.random_seed,
            save_games=config.evaluation.save_games,
            save_path=config.evaluation.save_path,
            log_level=config.evaluation.log_level,
            wandb_logging=config.evaluation.wandb_log_eval,
            update_elo=config.evaluation.update_elo,
            enable_in_memory_evaluation=config.evaluation.enable_in_memory_evaluation,
            model_weight_cache_size=config.evaluation.model_weight_cache_size,
            enable_parallel_execution=config.evaluation.enable_parallel_execution,
            process_restart_threshold=config.evaluation.process_restart_threshold,
            temp_agent_device=config.evaluation.temp_agent_device,
            clear_cache_after_evaluation=config.evaluation.clear_cache_after_evaluation,
            # Add strategy-specific parameters
            opponent_name=config.evaluation.opponent_type,
            strategy_params=getattr(config.evaluation, "strategy_params", {}),
        )
        self.evaluation_manager = EnhancedEvaluationManager(
            eval_config,
            self.run_name,
            pool_size=config.evaluation.previous_model_pool_size,
            elo_registry_path=config.evaluation.elo_registry_path,
            enable_background_tournaments=getattr(
                config.evaluation, "enable_background_tournaments", False
            ),
            enable_advanced_analytics=getattr(
                config.evaluation, "enable_advanced_analytics", False
            ),
            enable_enhanced_opponents=getattr(
                config.evaluation, "enable_enhanced_opponents", False
            ),
            analytics_output_dir=Path(self.model_dir) / "analytics",
        )
        self.evaluation_elo_snapshot: Optional[Dict[str, Any]] = None
        self.callback_manager = CallbackManager(config, self.model_dir)
        self.setup_manager = SetupManager(config, self.device)

        # Setup components using SetupManager
        self._initialize_components()

        # Configure evaluation manager with runtime components
        self.evaluation_manager.setup(
            device=config.env.device,
            policy_mapper=self.policy_output_mapper,
            model_dir=self.model_dir,
            wandb_active=self.is_train_wandb_active,
        )

        # Setup display and callbacks
        self.display = self.display_manager.setup_display(self)
        self.callbacks = self.callback_manager.setup_default_callbacks()

        # Initialize TrainingLoopManager
        self.training_loop_manager = TrainingLoopManager(trainer=self)

        # Variables for UI instrumentation
        self.last_gradient_norm: float = 0.0
        self.last_weight_updates: Dict[str, float] = {}

    def _initialize_components(self):
        """Initialize all training components using SetupManager."""
        # Setup game components
        (
            self.game,
            self.policy_output_mapper,
            self.action_space_size,
            self.obs_space_shape,
        ) = self.setup_manager.setup_game_components(
            self.env_manager, self.rich_console
        )

        # Setup training components
        self.model, self.agent, self.experience_buffer = (
            self.setup_manager.setup_training_components(self.model_manager)
        )

        # Setup step manager
        self.step_manager = self.setup_manager.setup_step_manager(
            self.game, self.agent, self.policy_output_mapper, self.experience_buffer
        )

        # Handle checkpoint resume
        self.resumed_from_checkpoint = self.setup_manager.handle_checkpoint_resume(
            self.model_manager,
            self.agent,
            self.model_dir,
            self.args.resume,
            self.metrics_manager,
            self.logger,
        )

    def _initialize_game_state(self, log_both) -> EpisodeState:
        """Initialize the game state for training using EnvManager."""
        try:
            self.env_manager.reset_game()
            if not self.step_manager:
                raise RuntimeError("StepManager not initialized")

            # Type narrowing: assert that step_manager is not None
            assert (
                self.step_manager is not None
            ), "StepManager should be initialized at this point"
            return self.step_manager.reset_episode()
        except (RuntimeError, ValueError, OSError) as e:
            log_both(
                f"CRITICAL: Error during initial game.reset(): {e}. Aborting.",
                also_to_wandb=True,
            )
            if self.is_train_wandb_active and wandb.run:
                wandb.finish(exit_code=1)
            raise RuntimeError(f"Game initialization error: {e}") from e

    def perform_ppo_update(self, current_obs_np, log_both):
        """Perform a PPO update."""
        if not self.agent or not self.experience_buffer:
            log_both(
                "[ERROR] PPO update called but agent or experience buffer is not initialized.",
                also_to_wandb=True,
            )
            return

        # Type narrowing: assert that agent and experience_buffer are not None
        assert self.agent is not None, "Agent should be initialized at this point"
        assert (
            self.experience_buffer is not None
        ), "Experience buffer should be initialized at this point"

        with torch.no_grad():
            last_value_pred_for_gae = self.agent.get_value(current_obs_np)

        self.experience_buffer.compute_advantages_and_returns(last_value_pred_for_gae)
        # Track weight updates by comparing state before and after learn()
        model_any: Any = getattr(self.agent, "model", None)
        old_state = (
            {
                name: p.detach().clone()
                for name, p in cast(Any, model_any).named_parameters()
            }
            if model_any is not None
            else {}
        )
        learn_metrics = self.agent.learn(self.experience_buffer)
        new_state = (
            {name: p.detach() for name, p in cast(Any, model_any).named_parameters()}
            if model_any is not None
            else {}
        )
        self.last_weight_updates = {
            name: (new_state[name] - old_state[name]).norm().item()
            for name in old_state.keys()
        }
        self.last_gradient_norm = getattr(self.agent, "last_gradient_norm", 0.0)
        self.experience_buffer.clear()

        # Format PPO metrics for display using MetricsManager
        ppo_metrics_display = self.metrics_manager.format_ppo_metrics(learn_metrics)
        self.metrics_manager.update_progress_metrics("ppo_metrics", ppo_metrics_display)

        # Format detailed metrics for logging
        ppo_metrics_log = self.metrics_manager.format_ppo_metrics_for_logging(
            learn_metrics
        )

        log_both(
            f"PPO Update @ ts {self.metrics_manager.global_timestep+1}. Metrics: {ppo_metrics_log}",
            also_to_wandb=True,
            wandb_data=learn_metrics,
        )

    def _log_run_info(self, log_both):
        """Log run information at the start of training using SetupManager."""
        self.setup_manager.log_run_info(
            self.session_manager,
            self.model_manager,
            self.agent,
            self.metrics_manager,
            log_both,
        )

    def _finalize_training(self, log_both):
        """Finalize training and save final model and checkpoint via ModelManager."""
        log_both(
            f"Training loop finished at timestep {self.metrics_manager.global_timestep}. Total episodes: {self.metrics_manager.total_episodes_completed}.",
            also_to_wandb=True,
        )

        if not self.agent:
            log_both(
                "[ERROR] Finalize training: Agent not initialized. Cannot save model or checkpoint.",
                also_to_wandb=True,
            )
            if (
                self.is_train_wandb_active and wandb.run
            ):  # Ensure WandB is finalized if active
                self.session_manager.finalize_session()
                log_both("Weights & Biases run finished due to error.")
            return

        game_stats = self.metrics_manager.get_final_stats()

        if self.metrics_manager.global_timestep >= self.config.training.total_timesteps:
            log_both(
                "Training successfully completed all timesteps. Saving final model.",
                also_to_wandb=True,
            )
            success, final_model_path = self.model_manager.save_final_model(
                agent=self.agent,
                model_dir=self.model_dir,
                global_timestep=self.metrics_manager.global_timestep,
                total_episodes_completed=self.metrics_manager.total_episodes_completed,
                game_stats=game_stats,
                run_name=self.run_name,
                is_wandb_active=self.is_train_wandb_active,
            )
            if success and final_model_path:
                log_both(
                    f"Final model processing (save & artifact) successful: {final_model_path}",
                    also_to_wandb=True,
                )
            else:
                log_both(
                    f"[ERROR] Failed to save/artifact final model for timestep {self.metrics_manager.global_timestep}.",
                    also_to_wandb=True,
                )

            # WandB finishing is handled by SessionManager or after all save attempts
        else:
            log_both(
                f"[WARNING] Training interrupted at timestep {self.metrics_manager.global_timestep} (before {self.config.training.total_timesteps} total).",
                also_to_wandb=True,
            )

        # Always attempt to save a final checkpoint
        log_both(
            f"Attempting to save final checkpoint at timestep {self.metrics_manager.global_timestep}.",
            also_to_wandb=False,
        )
        ckpt_success, final_ckpt_path = self.model_manager.save_final_checkpoint(
            agent=self.agent,
            model_dir=self.model_dir,
            global_timestep=self.metrics_manager.global_timestep,
            total_episodes_completed=self.metrics_manager.total_episodes_completed,
            game_stats=game_stats,
            run_name=self.run_name,
            is_wandb_active=self.is_train_wandb_active,
        )
        if ckpt_success and final_ckpt_path:
            log_both(
                f"Final checkpoint processing (save & artifact) successful: {final_ckpt_path}",
                also_to_wandb=True,
            )
        elif (
            self.metrics_manager.global_timestep > 0
        ):  # Only log error if a checkpoint was expected
            log_both(
                f"[ERROR] Failed to save/artifact final checkpoint for timestep {self.metrics_manager.global_timestep}.",
                also_to_wandb=True,
            )

        if self.is_train_wandb_active and wandb.run:
            self.session_manager.finalize_session()
            log_both("Weights & Biases run finished.")

        # Finalize display and save console output
        self.display_manager.finalize_display(self.run_name, self.run_artifact_dir)
        
        # Cleanup WebUI if enabled
        if self.webui_manager:
            self.webui_manager.stop()
        if self.webui_http_server:
            self.webui_http_server.stop()

    def run_training_loop(self):
        """Executes the main training loop by delegating to TrainingLoopManager."""
        self.session_manager.log_session_start()

        with TrainingLogger(
            self.log_file_path,
            rich_console=self.rich_console,
            rich_log_panel=self.rich_log_messages,
            also_stdout=False,
        ) as logger:

            def log_both_impl(
                message: str,
                also_to_wandb: bool = False,
                wandb_data: Optional[Dict] = None,
                log_level: str = "info",
            ):
                logger.log(message)
                if self.is_train_wandb_active and also_to_wandb and wandb.run:
                    log_payload = {"train_message": message}
                    if wandb_data:
                        log_payload.update(wandb_data)
                    wandb.log(log_payload, step=self.metrics_manager.global_timestep)

            self.log_both = log_both_impl

            session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_both(
                f"--- SESSION START: {self.run_name} at {session_start_time} ---"
            )

            # Setup run information logging
            self._log_run_info(self.log_both)

            # Log checkpoint resume status
            if self.resumed_from_checkpoint:
                self.log_both(
                    f"Resumed training from checkpoint: {self.resumed_from_checkpoint}"
                )

            initial_episode_state = self._initialize_game_state(self.log_both)
            self.training_loop_manager.set_initial_episode_state(initial_episode_state)

            # Start the Rich Live display as a context manager
            with self.display.start():
                try:
                    self.training_loop_manager.run()
                except KeyboardInterrupt:
                    self.log_both(
                        "Trainer caught KeyboardInterrupt from TrainingLoopManager. Finalizing.",
                        also_to_wandb=True,
                    )
                except (RuntimeError, ValueError, AttributeError, ImportError) as e:
                    self.log_both(
                        f"Trainer caught unhandled exception from TrainingLoopManager: {e}. Finalizing.",
                        also_to_wandb=True,
                    )
                finally:
                    self._finalize_training(self.log_both)

    def _handle_checkpoint_resume(self):
        """
        Handle checkpoint resume for backward compatibility.

        This method delegates to SetupManager for consistency with the refactored architecture.
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized before _handle_checkpoint_resume")

        # Delegate to SetupManager
        result = self.setup_manager.handle_checkpoint_resume(
            self.model_manager,
            self.agent,
            self.model_dir,
            self.args.resume,
            self.metrics_manager,
            self.logger,
        )

        # Update trainer's state to match the result
        self.resumed_from_checkpoint = result

        return result
