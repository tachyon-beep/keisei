# keisei/training/training_loop_manager.py
"""
Manages the main training loop execution, previously part of the Trainer class.
"""
import time
import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import torch.nn as nn

from keisei.shogi.shogi_core_definitions import Color
from keisei.utils.unified_logger import log_info_to_stderr

# Constants
STEP_MANAGER_NOT_AVAILABLE_MSG = "StepManager is not available"

if TYPE_CHECKING:
    from typing import Callable  # Added Callable

    from keisei.config_schema import AppConfig
    from keisei.core.experience_buffer import ExperienceBuffer
    from keisei.core.ppo_agent import PPOAgent
    from keisei.training.callbacks import Callback
    from keisei.training.display import TrainingDisplay
    from keisei.training.parallel import ParallelManager
    from keisei.training.step_manager import (  # Added StepResult
        EpisodeState,
        StepManager,
        StepResult,
    )
    from keisei.training.trainer import Trainer  # Forward reference


class TrainingLoopManager:
    """
    Manages the primary iteration logic of the training loop.
    """

    def __init__(
        self,
        trainer: "Trainer",
        # Components are accessed via trainer instance
    ):
        self.trainer = trainer
        self.config = trainer.config  # Convenience access
        self.agent = trainer.agent  # Convenience access
        self.buffer = trainer.experience_buffer  # Convenience access
        self.step_manager = trainer.step_manager  # Convenience access
        self.display = trainer.display  # Convenience access
        self.callbacks = trainer.callbacks  # Convenience access

        self.current_epoch: int = 0
        self.episode_state: Optional["EpisodeState"] = None  # Set by Trainer before run

        # For SPS calculation and display update throttling
        self.last_time_for_sps: float = 0.0
        self.steps_since_last_time_for_sps: int = 0
        self.last_display_update_time: float = 0.0

        # Initialize parallel manager if enabled
        self.parallel_manager: Optional["ParallelManager"] = None
        if self.config.parallel.enabled:
            from keisei.training.parallel import ParallelManager

            # Build config dictionaries for parallel manager
            env_config = self._build_env_config()
            model_config = self._build_model_config()

            self.parallel_manager = ParallelManager(
                env_config=env_config,
                model_config=model_config,
                parallel_config=self.config.parallel.model_dump(),
                device=self.config.env.device,
            )

    def set_initial_episode_state(self, initial_episode_state: "EpisodeState"):
        """Sets the initial episode state, typically provided by the Trainer."""
        self.episode_state = initial_episode_state

    def run(self):
        """
        Executes the main training loop.
        This method contains the core iteration logic.
        """
        log_both = self.trainer.log_both
        if not log_both:
            # This should be set by Trainer.run_training_loop before calling this
            raise RuntimeError(
                "Trainer's log_both callback is not set before running TrainingLoopManager."
            )
        if self.episode_state is None:
            raise RuntimeError(
                "Initial episode state not set in TrainingLoopManager before run."
            )

        self.last_time_for_sps = time.time()
        self.steps_since_last_time_for_sps = 0
        self.last_display_update_time = time.time()

        # Start parallel workers if parallel training is enabled
        if self.parallel_manager and self.config.parallel.enabled:
            if self.trainer.agent and self.trainer.agent.model:
                log_both(
                    f"Starting {self.config.parallel.num_workers} parallel workers..."
                )
                if self.parallel_manager.start_workers(
                    cast(nn.Module, self.trainer.agent.model)
                ):
                    log_both("Parallel workers started successfully")
                else:
                    log_both(
                        "Failed to start parallel workers, falling back to sequential training"
                    )
                    self.parallel_manager = None
            else:
                log_both("Cannot start parallel workers: model not available")
                self.parallel_manager = None

        try:
            while (
                self.trainer.metrics_manager.global_timestep
                < self.config.training.total_timesteps
            ):
                self.current_epoch += 1

                self._run_epoch(log_both)

                if (
                    self.trainer.metrics_manager.global_timestep
                    >= self.config.training.total_timesteps
                ):
                    log_both(
                        f"Target timesteps ({self.config.training.total_timesteps}) reached during epoch {self.current_epoch}."
                    )
                    break

                if self.episode_state and self.episode_state.current_obs is not None:
                    self.trainer.metrics_manager.set_processing(True)
                    try:
                        self.trainer.perform_ppo_update(
                            self.episode_state.current_obs, log_both
                        )
                    finally:
                        self.trainer.metrics_manager.set_processing(False)
                else:
                    log_both(
                        "[WARNING] Skipping PPO update due to missing current_obs in episode_state. "
                        f"(Timestep: {self.trainer.metrics_manager.global_timestep})",
                        also_to_wandb=True,
                    )

                # Execute step callbacks using centralized callback manager with error handling
                self.trainer.callback_manager.execute_step_callbacks(self.trainer)

                # TODO: Async callbacks are silently dropped in this sync path.
                # If a caller registers async callbacks (via use_async_evaluation())
                # and then enters run() instead of run_async(), those callbacks will
                # never execute. Consider raising an error at the top of run() when
                # async callbacks are registered, or bridging them with a dedicated
                # thread-based event loop. Currently no production caller hits this
                # because use_async_evaluation() is only enabled alongside run_async().
                # See also: run_async() which properly awaits async callbacks.

        except KeyboardInterrupt:
            log_both(
                "Training interrupted by user (KeyboardInterrupt in TrainingLoopManager).",
                also_to_wandb=True,
            )
            raise
        except (RuntimeError, ValueError, AttributeError) as e:
            log_message = f"Training error in TrainingLoopManager.run: {e}"
            if hasattr(self.trainer, "logger") and self.trainer.logger:
                self.trainer.logger.log(log_message)
            else:
                log_info_to_stderr("TrainingLoopManager", log_message)
            log_both(f"Training error in training loop: {e}", also_to_wandb=True)
            raise

    async def run_async(self):
        """
        Async version of the training loop for full async callback support.
        This is an alternative to run() that supports native async callback execution.
        """
        log_both = self.trainer.log_both
        if not log_both:
            raise RuntimeError(
                "Trainer's log_both callback is not set before running TrainingLoopManager."
            )
        if self.episode_state is None:
            raise RuntimeError(
                "Initial episode state not set in TrainingLoopManager before run."
            )

        self.last_time_for_sps = time.time()
        self.steps_since_last_time_for_sps = 0
        self.last_display_update_time = time.time()

        # Start parallel workers if parallel training is enabled
        if self.parallel_manager and self.config.parallel.enabled:
            if self.trainer.agent and self.trainer.agent.model:
                log_both(
                    f"Starting {self.config.parallel.num_workers} parallel workers..."
                )
                if self.parallel_manager.start_workers(
                    cast(nn.Module, self.trainer.agent.model)
                ):
                    log_both("Parallel workers started successfully")
                else:
                    log_both(
                        "Failed to start parallel workers, falling back to sequential training"
                    )
                    self.parallel_manager = None
            else:
                log_both("Cannot start parallel workers: model not available")
                self.parallel_manager = None

        try:
            while (
                self.trainer.metrics_manager.global_timestep
                < self.config.training.total_timesteps
            ):
                self.current_epoch += 1

                self._run_epoch(log_both)

                if (
                    self.trainer.metrics_manager.global_timestep
                    >= self.config.training.total_timesteps
                ):
                    log_both(
                        f"Target timesteps ({self.config.training.total_timesteps}) reached during epoch {self.current_epoch}."
                    )
                    break

                if self.episode_state and self.episode_state.current_obs is not None:
                    self.trainer.metrics_manager.set_processing(True)
                    try:
                        self.trainer.perform_ppo_update(
                            self.episode_state.current_obs, log_both
                        )
                    finally:
                        self.trainer.metrics_manager.set_processing(False)
                else:
                    log_both(
                        "[WARNING] Skipping PPO update due to missing current_obs in episode_state. "
                        f"(Timestep: {self.trainer.metrics_manager.global_timestep})",
                        also_to_wandb=True,
                    )

                # Execute sync step callbacks first
                self.trainer.callback_manager.execute_step_callbacks(self.trainer)

                # Execute async callbacks natively
                if self.trainer.callback_manager.has_async_callbacks():
                    async_metrics = await self.trainer.callback_manager.execute_step_callbacks_async(
                        self.trainer
                    )
                    if async_metrics:
                        # Integrate async callback metrics into training metrics
                        self.trainer.metrics_manager.pending_progress_updates.update(
                            async_metrics
                        )

        except KeyboardInterrupt:
            log_both(
                "Training interrupted by user (KeyboardInterrupt in TrainingLoopManager).",
                also_to_wandb=True,
            )
            raise
        except (RuntimeError, ValueError, AttributeError) as e:
            log_message = f"Training error in TrainingLoopManager.run_async: {e}"
            if hasattr(self.trainer, "logger") and self.trainer.logger:
                self.trainer.logger.log(log_message)
            else:
                log_info_to_stderr("TrainingLoopManager", log_message)
            log_both(f"Training error in async training loop: {e}", also_to_wandb=True)
            raise

    def _run_epoch(self, log_both):
        """
        Runs a single epoch, collecting experiences until the buffer is full or total timesteps are met.
        Uses parallel collection if enabled, otherwise falls back to sequential collection.
        """
        num_steps_collected_this_epoch = 0

        # Check if parallel training is enabled
        if self.parallel_manager and self.config.parallel.enabled:
            # Parallel experience collection
            num_steps_collected_this_epoch = self._run_epoch_parallel(log_both)
        else:
            # Sequential experience collection (existing logic)
            num_steps_collected_this_epoch = self._run_epoch_sequential(log_both)

        # Update metrics regardless of collection mode
        self.trainer.metrics_manager.pending_progress_updates.setdefault(
            "steps_collected_this_epoch", num_steps_collected_this_epoch
        )

    def _run_epoch_parallel(self, log_both):
        """
        Parallel experience collection using worker processes.
        """
        if not self.parallel_manager:
            log_both("ParallelManager not available, falling back to sequential mode.")
            return self._run_epoch_sequential(log_both)

        num_steps_collected = 0
        collection_attempts = 0
        max_collection_attempts = 50  # Prevent infinite loops

        log_both(
            f"Starting parallel experience collection for epoch {self.current_epoch}"
        )

        while (
            num_steps_collected < self.config.training.steps_per_epoch
            and self.trainer.metrics_manager.global_timestep
            < self.config.training.total_timesteps
            and collection_attempts < max_collection_attempts
        ):

            collection_attempts += 1

            # Synchronize model with workers if needed
            if (
                self.agent
                and self.agent.model
                and self.parallel_manager.sync_model_if_needed(
                    cast(nn.Module, self.agent.model),
                    self.trainer.metrics_manager.global_timestep,
                )
            ):
                log_both(
                    f"Model synchronized with workers at step {self.trainer.metrics_manager.global_timestep}"
                )

            # Collect experiences from workers
            try:
                if self.buffer:
                    experiences_collected = self.parallel_manager.collect_experiences(
                        self.buffer
                    )

                    if experiences_collected > 0:
                        num_steps_collected += experiences_collected
                        self.trainer.metrics_manager.increment_timestep_by(
                            experiences_collected
                        )
                        # Fix B11: Update SPS calculation counter for parallel mode
                        self.steps_since_last_time_for_sps += experiences_collected

                        log_both(
                            f"Collected {experiences_collected} experiences from workers "
                            f"(total this epoch: {num_steps_collected})",
                            also_to_wandb=False,
                        )

                        # Update display periodically
                        if num_steps_collected % 100 == 0:  # Every 100 steps
                            self._update_display_progress(num_steps_collected)
                    else:
                        # No experiences collected this round, brief wait
                        time.sleep(0.01)  # 10ms wait
                else:
                    log_both("Experience buffer not available for parallel collection")
                    break

            except (
                RuntimeError,
                ValueError,
                AttributeError,
                ConnectionError,
                TimeoutError,
            ) as e:
                log_both(
                    f"Error collecting parallel experiences: {e}. "
                    f"Attempt {collection_attempts}/{max_collection_attempts}",
                    also_to_wandb=True,
                )
                if collection_attempts >= max_collection_attempts:
                    log_both(
                        "Max collection attempts reached. Falling back to sequential mode.",
                        also_to_wandb=True,
                    )
                    return self._run_epoch_sequential(log_both)

        log_both(
            f"Parallel epoch {self.current_epoch} completed. "
            f"Collected {num_steps_collected} experiences in {collection_attempts} attempts."
        )

        return num_steps_collected

    def _run_epoch_sequential(self, log_both: "Callable"):
        """
        Sequential experience collection.
        Refactored for clarity and reduced complexity.
        """
        num_steps_collected_this_epoch = 0
        while num_steps_collected_this_epoch < self.config.training.steps_per_epoch:
            should_continue = self._process_step_and_handle_episode(log_both)
            if not should_continue:
                break

            num_steps_collected_this_epoch += 1
            self._handle_display_updates()

        return num_steps_collected_this_epoch

    def _handle_successful_step(
        self,
        episode_state: "EpisodeState",
        step_result: "StepResult",  # Corrected type hint
        log_both: "Callable",  # Corrected type hint
    ) -> "EpisodeState":
        """Handles the logic after a successful step, including episode end."""
        if self.step_manager is None:
            raise RuntimeError(STEP_MANAGER_NOT_AVAILABLE_MSG)

        updated_episode_state = self.step_manager.update_episode_state(
            episode_state, step_result
        )

        if step_result.done:
            current_cumulative_stats = {
                "black_wins": self.trainer.metrics_manager.black_wins,
                "white_wins": self.trainer.metrics_manager.white_wins,
                "draws": self.trainer.metrics_manager.draws,
            }

            new_episode_state_after_end, episode_winner_color = (
                self.step_manager.handle_episode_end(
                    updated_episode_state,
                    step_result,
                    current_cumulative_stats,
                    self.trainer.metrics_manager.total_episodes_completed,
                    log_both,
                )
            )

            winner_color_enum = (
                Color.BLACK
                if episode_winner_color == "black"
                else Color.WHITE if episode_winner_color == "white" else None
            )
            self.trainer.metrics_manager.update_episode_stats(winner_color_enum)

            result_str = "win" if episode_winner_color in ("black", "white") else "draw"
            self._log_episode_metrics(updated_episode_state, result_str)
            return new_episode_state_after_end
        else:
            return updated_episode_state

    def _process_step_and_handle_episode(self, log_both: "Callable") -> bool:
        """Processes a single step and handles episode completion if necessary.
        Returns True if the loop should continue, False if a critical error occurred or max timesteps reached.
        """
        if (
            self.trainer.metrics_manager.global_timestep
            >= self.config.training.total_timesteps
        ):
            return False  # Stop epoch

        if self.episode_state is None:
            log_both("[ERROR] Episode state is None. Resetting.", also_to_wandb=True)
            if self.step_manager is None:
                raise RuntimeError(STEP_MANAGER_NOT_AVAILABLE_MSG)
            self.episode_state = self.step_manager.reset_episode()
            if self.episode_state is None:
                raise RuntimeError("Failed to reset episode_state.")
            return True  # Continue epoch

        if self.step_manager is None:
            raise RuntimeError(STEP_MANAGER_NOT_AVAILABLE_MSG)

        step_result = self.step_manager.execute_step(
            episode_state=self.episode_state,
            global_timestep=self.trainer.metrics_manager.global_timestep,
            logger_func=log_both,
        )

        if not step_result.success:
            log_both(
                f"[WARNING] Step failed at {self.trainer.metrics_manager.global_timestep}. Resetting.",
                also_to_wandb=True,
            )
            if self.step_manager is None:
                raise RuntimeError(STEP_MANAGER_NOT_AVAILABLE_MSG)
            self.episode_state = self.step_manager.reset_episode()
            return True  # Continue epoch

        self.episode_state = self._handle_successful_step(
            self.episode_state, step_result, log_both
        )

        self.trainer.metrics_manager.increment_timestep()
        self.steps_since_last_time_for_sps += 1
        return True  # Continue epoch

    def _log_episode_metrics(self, episode_state: "EpisodeState", result: str):
        """Logs metrics at the end of an episode."""
        ep_len = episode_state.episode_length
        ep_rew = episode_state.episode_reward
        ep_metrics_str = f"L:{ep_len} R:{ep_rew:.2f}"
        turns_count = ep_len
        self.trainer.metrics_manager.log_episode_metrics(
            ep_len,
            turns_count,
            result,
            ep_rew,
            (
                self.trainer.step_manager.move_history
                if self.trainer.step_manager
                else None
            ),
            self.trainer.policy_output_mapper,
        )

        total_games = (
            self.trainer.metrics_manager.black_wins
            + self.trainer.metrics_manager.white_wins
            + self.trainer.metrics_manager.draws
        )
        bw_rate = (
            self.trainer.metrics_manager.black_wins / total_games
            if total_games > 0
            else 0.0
        )
        ww_rate = (
            self.trainer.metrics_manager.white_wins / total_games
            if total_games > 0
            else 0.0
        )
        d_rate = (
            self.trainer.metrics_manager.draws / total_games if total_games > 0 else 0.0
        )

        self.trainer.metrics_manager.pending_progress_updates.update(
            {
                "ep_metrics": ep_metrics_str,
                "black_wins_cum": self.trainer.metrics_manager.black_wins,
                "white_wins_cum": self.trainer.metrics_manager.white_wins,
                "draws_cum": self.trainer.metrics_manager.draws,
                "black_win_rate": bw_rate,
                "white_win_rate": ww_rate,
                "draw_rate": d_rate,
                "total_episodes": self.trainer.metrics_manager.total_episodes_completed,
            }
        )

    def _update_display_if_needed(self, extra_updates: Optional[dict] = None):
        """
        Shared helper method to update display with current progress.

        Args:
            extra_updates: Optional dictionary of additional updates to include
        """
        current_time = time.time()
        display_update_interval = getattr(
            self.config.training, "rich_display_update_interval_seconds", 0.2
        )

        if (current_time - self.last_display_update_time) > display_update_interval:
            time_delta_sps = current_time - self.last_time_for_sps
            current_speed = (
                self.steps_since_last_time_for_sps / time_delta_sps
                if time_delta_sps > 0
                else 0.0
            )

            self.trainer.metrics_manager.pending_progress_updates.setdefault(
                "current_epoch", self.current_epoch
            )

            # Add any extra updates provided
            if extra_updates:
                for key, value in extra_updates.items():
                    self.trainer.metrics_manager.pending_progress_updates.setdefault(
                        key, value
                    )

            if hasattr(self.display, "update_progress") and callable(
                self.display.update_progress
            ):
                self.display.update_progress(
                    self.trainer,
                    current_speed,
                    self.trainer.metrics_manager.pending_progress_updates,
                )
            
            # Update WebUI if enabled (parallel to display update)
            if self.trainer.webui_manager:
                self.trainer.webui_manager.update_progress(
                    self.trainer,
                    current_speed,
                    self.trainer.metrics_manager.pending_progress_updates,
                )
                
            self.trainer.metrics_manager.pending_progress_updates.clear()

            self.last_time_for_sps = current_time
            self.steps_since_last_time_for_sps = 0
            self.last_display_update_time = current_time

    def _handle_display_updates(self):
        """Handles periodic display updates based on time and step intervals."""
        if (
            self.trainer.metrics_manager.global_timestep
            % self.config.training.render_every_steps
            == 0
        ):
            if hasattr(self.display, "refresh_dashboard_panels") and callable(
                self.display.refresh_dashboard_panels
            ):
                self.display.refresh_dashboard_panels(self.trainer)
            
            # Update WebUI dashboard if enabled (parallel to display refresh)
            if self.trainer.webui_manager:
                self.trainer.webui_manager.refresh_dashboard_panels(self.trainer)

        self._update_display_if_needed()

    def _build_env_config(self) -> Dict[str, Any]:
        """Build environment configuration dictionary for parallel workers."""
        return {
            "device": "cpu",  # Workers typically use CPU for environment simulation
            "input_channels": self.config.env.input_channels,
            "num_actions_total": self.config.env.num_actions_total,
            "seed": self.config.env.seed,
            "input_features": self.config.training.input_features,
        }

    def _build_model_config(self) -> Dict[str, Any]:
        """Build model configuration dictionary for parallel workers."""
        return {
            "model_type": self.config.training.model_type,
            "tower_depth": self.config.training.tower_depth,
            "tower_width": self.config.training.tower_width,
            "se_ratio": self.config.training.se_ratio,
            "obs_shape": (self.config.env.input_channels, 9, 9),  # Standard Shogi board
            "num_actions": self.config.env.num_actions_total,
        }

    def _update_display_progress(self, num_steps_collected):
        """Update display with current progress during parallel collection."""
        extra_updates = {"parallel_steps_collected": num_steps_collected}
        self._update_display_if_needed(extra_updates)
