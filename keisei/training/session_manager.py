"""
training/session_manager.py: Session lifecycle management for Shogi RL training.

This module handles session-level concerns including:
- Run name generation and validation
- Directory structure creation and management
- Configuration serialization and persistence
- WandB initialization and configuration
- Session logging and reporting
- Evaluation logging and WandB integration
"""

import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import wandb
from keisei.config_schema import AppConfig, EvaluationConfig
from keisei.utils.unified_logger import log_error_to_stderr, log_warning_to_stderr
from keisei.utils.utils import generate_run_name

from . import utils


class SessionManager:
    """Manages session-level lifecycle for training runs."""

    def __init__(self, config: AppConfig, args: Any, run_name: Optional[str] = None):
        """
        Initialize the SessionManager.

        Args:
            config: Application configuration
            args: Command-line arguments
            run_name: Optional explicit run name (overrides config and auto-generation)
        """
        self.config = config
        self.args = args

        # Determine run_name: explicit > CLI > config > auto-generate
        if run_name:
            self._run_name = run_name
        elif hasattr(args, "run_name") and args.run_name:
            self._run_name = args.run_name
        elif (
            hasattr(config, "logging")
            and hasattr(config.logging, "run_name")
            and config.logging.run_name
        ):
            self._run_name = config.logging.run_name
        else:
            self._run_name = generate_run_name(config, None)

        # Sanitize run_name to prevent path traversal
        self._run_name = self._sanitize_run_name(self._run_name)

        # Initialize paths - will be set by setup_directories()
        self._run_artifact_dir: Optional[str] = None
        self._model_dir: Optional[str] = None
        self._log_file_path: Optional[str] = None
        self._eval_log_file_path: Optional[str] = None

        # WandB state
        self._is_wandb_active: Optional[bool] = None

    @staticmethod
    def _sanitize_run_name(name: str) -> str:
        """Sanitize run name to prevent path traversal and invalid characters."""
        # Strip path separators and parent directory references
        name = os.path.basename(name)
        # Keep only alphanumeric, hyphens, underscores, dots
        name = re.sub(r"[^\w\-.]", "_", name)
        # Collapse multiple underscores/dots and strip leading dots
        name = re.sub(r"\.{2,}", ".", name)
        name = name.lstrip(".")
        return name or "unnamed_run"

    @property
    def run_name(self) -> str:
        """Get the run name."""
        return self._run_name

    @property
    def run_artifact_dir(self) -> str:
        """Get the run artifact directory path."""
        if self._run_artifact_dir is None:
            raise RuntimeError(
                "Directories not yet set up. Call setup_directories() first."
            )
        return self._run_artifact_dir

    @property
    def model_dir(self) -> str:
        """Get the model directory path."""
        if self._model_dir is None:
            raise RuntimeError(
                "Directories not yet set up. Call setup_directories() first."
            )
        return self._model_dir

    @property
    def log_file_path(self) -> str:
        """Get the log file path."""
        if self._log_file_path is None:
            raise RuntimeError(
                "Directories not yet set up. Call setup_directories() first."
            )
        return self._log_file_path

    @property
    def eval_log_file_path(self) -> str:
        """Get the evaluation log file path."""
        if self._eval_log_file_path is None:
            raise RuntimeError(
                "Directories not yet set up. Call setup_directories() first."
            )
        return self._eval_log_file_path

    @property
    def is_wandb_active(self) -> bool:
        """Check if WandB is active."""
        if self._is_wandb_active is None:
            raise RuntimeError("WandB not yet initialized. Call setup_wandb() first.")
        return bool(self._is_wandb_active)

    def setup_directories(self) -> Dict[str, str]:
        """
        Set up directory structure for the training run.

        Returns:
            Dictionary containing directory paths
        """
        try:
            dirs = utils.setup_directories(self.config, self._run_name)
            self._run_artifact_dir = dirs["run_artifact_dir"]
            self._model_dir = dirs["model_dir"]
            self._log_file_path = dirs["log_file_path"]
            self._eval_log_file_path = dirs["eval_log_file_path"]
            return dirs
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Failed to setup directories: {e}") from e

    def setup_wandb(self) -> bool:
        """
        Initialize Weights & Biases logging.

        Returns:
            True if WandB is active, False otherwise
        """
        if self._run_artifact_dir is None:
            raise RuntimeError("Directories must be set up before initializing WandB.")

        try:
            self._is_wandb_active = utils.setup_wandb(
                self.config, self._run_name, self._run_artifact_dir
            )
            return bool(self._is_wandb_active)
        except Exception as e:  # Catch all exceptions for WandB setup
            log_warning_to_stderr("SessionManager", f"WandB setup failed: {e}")
            self._is_wandb_active = False
            return False

    def setup_evaluation_logging(self, eval_config: EvaluationConfig) -> None:
        """
        Extend existing WandB session for evaluation.

        Args:
            eval_config: Evaluation configuration
        """
        if self._is_wandb_active and wandb.run:
            try:
                # Add evaluation-specific configuration to existing WandB session
                wandb.config.update(
                    {
                        "evaluation/strategy": eval_config.strategy,
                        "evaluation/num_games": eval_config.num_games,
                        "evaluation/max_concurrent_games": eval_config.max_concurrent_games,
                        "evaluation/opponent_type": eval_config.opponent_type,
                        "evaluation/enable_periodic_evaluation": eval_config.enable_periodic_evaluation,
                        "evaluation/evaluation_interval_timesteps": eval_config.evaluation_interval_timesteps,
                        "evaluation/enable_performance_monitoring": eval_config.enable_performance_monitoring,
                    }
                )

                # Add strategy-specific parameters if available
                if (
                    hasattr(eval_config, "strategy_params")
                    and eval_config.strategy_params
                ):
                    for key, value in eval_config.strategy_params.items():
                        wandb.config.update(
                            {f"evaluation/strategy_params/{key}": value}
                        )

            except Exception as e:
                log_warning_to_stderr(
                    "SessionManager",
                    f"Failed to update WandB config for evaluation: {e}",
                )

    def log_evaluation_metrics(self, result, step: int) -> None:
        """
        Log evaluation results to existing WandB session.

        Args:
            result: Evaluation result object
            step: Training step number
        """
        if self._is_wandb_active and wandb.run:
            try:
                metrics = {}

                # Extract summary statistics
                if hasattr(result, "summary_stats"):
                    stats = result.summary_stats

                    # Core evaluation metrics
                    if hasattr(stats, "win_rate"):
                        metrics["evaluation/win_rate"] = stats.win_rate
                    if hasattr(stats, "loss_rate"):
                        metrics["evaluation/loss_rate"] = stats.loss_rate
                    if hasattr(stats, "draw_rate"):
                        metrics["evaluation/draw_rate"] = stats.draw_rate
                    if hasattr(stats, "total_games"):
                        metrics["evaluation/total_games"] = stats.total_games
                    if hasattr(stats, "avg_game_length"):
                        metrics["evaluation/avg_game_length"] = stats.avg_game_length
                    if hasattr(stats, "avg_rewards"):
                        metrics["evaluation/avg_rewards"] = stats.avg_rewards

                # Performance metrics if available
                if hasattr(result, "performance_metrics"):
                    perf = result.performance_metrics
                    if hasattr(perf, "evaluation_latency_ms"):
                        metrics["evaluation/performance/latency_ms"] = (
                            perf.evaluation_latency_ms
                        )
                    if hasattr(perf, "memory_overhead_mb"):
                        metrics["evaluation/performance/memory_overhead_mb"] = (
                            perf.memory_overhead_mb
                        )
                    if hasattr(perf, "cpu_utilization_percent"):
                        metrics["evaluation/performance/cpu_utilization_percent"] = (
                            perf.cpu_utilization_percent
                        )
                    if (
                        hasattr(perf, "gpu_utilization_percent")
                        and perf.gpu_utilization_percent is not None
                    ):
                        metrics["evaluation/performance/gpu_utilization_percent"] = (
                            perf.gpu_utilization_percent
                        )

                # Game-level statistics if available
                if hasattr(result, "games") and result.games:
                    # Calculate additional statistics
                    game_lengths = [
                        len(game.moves) if hasattr(game, "moves") else 0
                        for game in result.games
                    ]
                    if game_lengths:
                        metrics["evaluation/min_game_length"] = min(game_lengths)
                        metrics["evaluation/max_game_length"] = max(game_lengths)

                    # Outcome distribution
                    outcomes = [
                        game.outcome
                        for game in result.games
                        if hasattr(game, "outcome")
                    ]
                    if outcomes:
                        wins = outcomes.count("win")
                        losses = outcomes.count("loss")
                        draws = outcomes.count("draw")
                        metrics["evaluation/wins"] = wins
                        metrics["evaluation/losses"] = losses
                        metrics["evaluation/draws"] = draws

                # Log all metrics
                if metrics:
                    wandb.log(metrics, step=step)

            except Exception as e:
                log_warning_to_stderr(
                    "SessionManager", f"Failed to log evaluation metrics to WandB: {e}"
                )

    def log_evaluation_performance(
        self, performance_metrics: Dict[str, float], step: int
    ) -> None:
        """
        Log evaluation performance metrics to WandB.

        Args:
            performance_metrics: Dictionary of performance metrics
            step: Training step number
        """
        if self._is_wandb_active and wandb.run:
            try:
                # Prefix performance metrics with evaluation namespace
                prefixed_metrics = {
                    f"evaluation/performance/{key}": value
                    for key, value in performance_metrics.items()
                }
                wandb.log(prefixed_metrics, step=step)

            except Exception as e:
                log_warning_to_stderr(
                    "SessionManager",
                    f"Failed to log evaluation performance to WandB: {e}",
                )

    def log_evaluation_sla_status(
        self, sla_passed: bool, violations: list, step: int
    ) -> None:
        """
        Log evaluation SLA status to WandB.

        Args:
            sla_passed: Whether SLA requirements were met
            violations: List of SLA violations
            step: Training step number
        """
        if self._is_wandb_active and wandb.run:
            try:
                metrics = {
                    "evaluation/sla/passed": sla_passed,
                    "evaluation/sla/violation_count": len(violations),
                }

                # Log individual violation types
                violation_types = set(v.split("=")[0] for v in violations if "=" in v)
                for violation_type in violation_types:
                    metrics[f"evaluation/sla/violations/{violation_type}"] = True

                wandb.log(metrics, step=step)

            except Exception as e:
                log_warning_to_stderr(
                    "SessionManager",
                    f"Failed to log evaluation SLA status to WandB: {e}",
                )

    def save_effective_config(self) -> None:
        """Save the effective configuration to a JSON file."""
        if self._run_artifact_dir is None:
            raise RuntimeError("Directories must be set up before saving config.")

        try:
            # Ensure the directory exists
            os.makedirs(self._run_artifact_dir, exist_ok=True)

            effective_config_str = utils.serialize_config(self.config)
            config_path = os.path.join(self._run_artifact_dir, "effective_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(effective_config_str)
        except (OSError, TypeError) as e:
            log_error_to_stderr(
                "SessionManager", f"Error saving effective_config.json: {e}"
            )
            raise RuntimeError(f"Failed to save effective config: {e}") from e

    def log_session_info(
        self,
        logger_func: Callable[[str], None],
        agent_info: Optional[Dict[str, Any]] = None,
        resumed_from_checkpoint: Optional[str] = None,
        global_timestep: int = 0,
        total_episodes_completed: int = 0,
    ) -> None:
        """
        Log comprehensive session information.

        Args:
            logger_func: Function to call for logging messages
            agent_info: Optional agent information (name, type)
            resumed_from_checkpoint: Optional checkpoint path if resumed
            global_timestep: Current global timestep
            total_episodes_completed: Total episodes completed so far
        """
        # Session title with optional WandB URL
        run_title = f"Keisei Training Run: {self._run_name}"
        if self._is_wandb_active and wandb.run and hasattr(wandb.run, "url"):
            run_title += f" (W&B: {wandb.run.url})"

        logger_func(run_title)
        logger_func(f"Run directory: {self.run_artifact_dir}")

        config_path = os.path.join(self.run_artifact_dir, "effective_config.json")
        logger_func(f"Effective config saved to: {config_path}")

        # Configuration information
        if self.config.env.seed is not None:
            logger_func(f"Random seed: {self.config.env.seed}")

        logger_func(f"Device: {self.config.env.device}")

        # Agent information
        if agent_info:
            logger_func(
                f"Agent: {agent_info.get('type', 'Unknown')} ({agent_info.get('name', 'Unknown')})"
            )

        logger_func(
            f"Total timesteps: {self.config.training.total_timesteps}, "
            f"Steps per PPO epoch: {self.config.training.steps_per_epoch}"
        )

        # Resume information
        if global_timestep > 0:
            if resumed_from_checkpoint:
                logger_func(
                    f"Resumed training from checkpoint: {resumed_from_checkpoint}"
                )
            logger_func(
                f"Resuming from timestep {global_timestep}, {total_episodes_completed} episodes completed."
            )
        else:
            logger_func("Starting fresh training.")

    def log_session_start(self) -> None:
        """Log session start event to file."""
        if self._log_file_path is None:
            raise RuntimeError(
                "Directories must be set up before logging session start."
            )

        try:
            with open(self._log_file_path, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] --- SESSION START: {self._run_name} ---\n")
        except (OSError, IOError) as e:
            log_error_to_stderr("SessionManager", f"Failed to log session start: {e}")

    def finalize_session(self) -> None:
        """Finalize the training session."""
        if self._is_wandb_active and wandb.run:
            import threading

            thread = threading.Thread(target=wandb.finish, daemon=True)
            thread.start()
            thread.join(timeout=10.0)

            if thread.is_alive():
                log_warning_to_stderr(
                    "SessionManager", "WandB finalization timed out"
                )

    def setup_seeding(self) -> None:
        """Setup random seeding based on configuration."""
        utils.setup_seeding(self.config)

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the session configuration.

        Returns:
            Dictionary containing session summary information
        """
        return {
            "run_name": self._run_name,
            "run_artifact_dir": self._run_artifact_dir,
            "model_dir": self._model_dir,
            "log_file_path": self._log_file_path,
            "eval_log_file_path": self._eval_log_file_path,
            "is_wandb_active": self._is_wandb_active,
            "seed": self.config.env.seed if hasattr(self.config.env, "seed") else None,
            "device": (
                self.config.env.device if hasattr(self.config.env, "device") else None
            ),
        }
