"""
training/model_manager.py: Model lifecycle management for Shogi RL training.

This module handles model-related concerns including:
- Model configuration and factory instantiation
- Mixed precision training setup
- torch.compile optimization with validation and fallback
- Checkpoint loading and resuming
- Model artifact creation for WandB
- Final model saving and persistence
- Performance benchmarking and optimization validation
"""

import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.cuda.amp import GradScaler

import wandb
from keisei.config_schema import AppConfig
from keisei.core.actor_critic_protocol import ActorCriticProtocol
from keisei.core.ppo_agent import PPOAgent
from keisei.shogi import features
from keisei.training.models import model_factory
from keisei.utils.performance_benchmarker import (
    PerformanceBenchmarker,
    create_benchmarker,
)
from keisei.utils.compilation_validator import (
    CompilationValidator,
    safe_compile_model,
    CompilationResult,
)

from . import utils


class ModelManager:
    """
    Manages model lifecycle for training runs with torch.compile optimization.

    New features in this version:
    - torch.compile integration with automatic fallback
    - Performance benchmarking and validation
    - Numerical equivalence verification
    - Compilation status tracking and reporting
    """

    def __init__(
        self, config: AppConfig, args: Any, device: torch.device, logger_func=None
    ):
        """
        Initialize the ModelManager.

        Args:
            config: Application configuration
            args: Command-line arguments
            device: PyTorch device for model operations
            logger_func: Optional logging function for status messages
        """
        self.config = config
        self.args = args
        self.device = device
        self.logger_func = logger_func or (lambda msg: None)

        # Initialize scaler early to satisfy type checker
        self.scaler: Optional[GradScaler] = None

        # Model configuration from args or config
        self.input_features = (
            getattr(args, "input_features", None) or config.training.input_features
        )
        self.model_type = getattr(args, "model", None) or config.training.model_type
        self.tower_depth = (
            getattr(args, "tower_depth", None) or config.training.tower_depth
        )
        self.tower_width = (
            getattr(args, "tower_width", None) or config.training.tower_width
        )
        self.se_ratio = getattr(args, "se_ratio", None) or config.training.se_ratio

        # Initialize feature spec and observation shape
        self._setup_feature_spec()

        # Initialize mixed precision
        self._setup_mixed_precision()

        # Initialize torch.compile optimization infrastructure
        self._setup_compilation_infrastructure()

        # Model will be created by explicit call to create_model()
        self.model: Optional[ActorCriticProtocol] = None

        # Track checkpoint resume status
        self.resumed_from_checkpoint: Optional[str] = None
        self.checkpoint_data: Optional[Dict[str, Any]] = None

        # Track compilation status
        self.compilation_result: Optional[CompilationResult] = None
        self.model_is_compiled: bool = False

    def _setup_feature_spec(self):
        """Setup feature specification and observation shape."""
        self.feature_spec = features.FEATURE_SPECS[self.input_features]
        self.obs_shape = (self.feature_spec.num_planes, 9, 9)

    def _setup_mixed_precision(self):
        """Setup mixed precision training if enabled and supported."""
        self.use_mixed_precision = (
            self.config.training.mixed_precision and self.device.type == "cuda"
        )

        if self.use_mixed_precision:
            self.scaler = GradScaler()
            self.logger_func("Mixed precision training enabled (CUDA).")
        elif self.config.training.mixed_precision and self.device.type != "cuda":
            self.logger_func(
                "Mixed precision training requested but CUDA is not available/selected. "
                "Proceeding without mixed precision."
            )
            self.use_mixed_precision = False
            self.scaler = None
        else:
            self.scaler = None

    def _setup_compilation_infrastructure(self):
        """Setup torch.compile optimization infrastructure."""
        # Initialize performance benchmarker if enabled
        if getattr(self.config.training, "enable_compilation_benchmarking", True):
            self.benchmarker = create_benchmarker(
                self.config.training, self.logger_func
            )
        else:
            self.benchmarker = None

        # Initialize compilation validator
        self.compilation_validator = CompilationValidator(
            config_training=self.config.training,
            logger_func=self.logger_func,
            benchmarker=self.benchmarker,
        )

        # Log compilation configuration
        if getattr(self.config.training, "enable_torch_compile", True):
            compile_mode = getattr(
                self.config.training, "torch_compile_mode", "default"
            )
            self.logger_func(
                f"torch.compile optimization enabled (mode: {compile_mode})"
            )
        else:
            self.logger_func("torch.compile optimization disabled")

    def create_model(self) -> ActorCriticProtocol:
        """
        Create the model using the model factory and apply torch.compile optimization.

        Returns:
            ActorCriticProtocol: Optimized model ready for training

        Raises:
            RuntimeError: If model creation fails
        """
        # Call model_factory once and assign to a temporary variable
        created_model = model_factory(
            model_type=self.model_type,
            obs_shape=self.obs_shape,
            num_actions=self.config.env.num_actions_total,
            tower_depth=self.tower_depth,
            tower_width=self.tower_width,
            se_ratio=self.se_ratio if self.se_ratio > 0 else None,
        )

        if not created_model:  # Check if model_factory returned a valid model
            raise RuntimeError("Model factory returned None, failed to create model.")

        # Move to device first
        created_model = created_model.to(self.device)

        if created_model is None:  # Should not happen if .to() raises on error
            raise RuntimeError("Failed to create model or move model to device.")

        # Apply torch.compile optimization if enabled
        optimized_model = self._apply_torch_compile_optimization(created_model)

        # Assign to self.model after optimization
        self.model = optimized_model

        return self.model

    def _apply_torch_compile_optimization(
        self, model: ActorCriticProtocol
    ) -> ActorCriticProtocol:
        """
        Apply torch.compile optimization with validation and fallback.

        Args:
            model: Model to optimize

        Returns:
            Optimized model (compiled or fallback to original)
        """
        # Create sample input for validation
        sample_input = torch.randn(
            1, *self.obs_shape, device=self.device, dtype=torch.float32
        )

        # Attempt compilation with validation
        compiled_model, compilation_result = safe_compile_model(
            model=model,
            sample_input=sample_input,
            config_training=self.config.training,
            logger_func=self.logger_func,
            benchmarker=self.benchmarker,
            model_name=f"{self.model_type}_model",
        )

        # Store compilation results for status reporting
        self.compilation_result = compilation_result
        self.model_is_compiled = (
            compilation_result.success and not compilation_result.fallback_used
        )

        # Log compilation status
        if self.model_is_compiled:
            perf_info = ""
            if compilation_result.performance_improvement:
                perf_info = (
                    f" ({compilation_result.performance_improvement:.2f}x speedup)"
                )
            self.logger_func(f"Model compilation successful{perf_info}")
        elif compilation_result.fallback_used:
            self.logger_func(
                f"Model compilation failed, using fallback: {compilation_result.error_message}"
            )
        else:
            self.logger_func("Model compilation skipped (disabled in configuration)")

        return compiled_model

    def get_compilation_status(self) -> Dict[str, Any]:
        """Get detailed compilation status information."""
        if not self.compilation_result:
            return {
                "attempted": False,
                "enabled": getattr(self.config.training, "enable_torch_compile", True),
                "message": "Compilation not attempted yet",
            }

        return {
            "attempted": True,
            "enabled": getattr(self.config.training, "enable_torch_compile", True),
            "success": self.compilation_result.success,
            "compiled": self.model_is_compiled,
            "fallback_used": self.compilation_result.fallback_used,
            "validation_passed": self.compilation_result.validation_passed,
            "performance_improvement": self.compilation_result.performance_improvement,
            "error_message": self.compilation_result.error_message,
            "metadata": self.compilation_result.metadata,
        }

    def benchmark_model_performance(
        self, model: Optional[ActorCriticProtocol] = None, num_iterations: int = 100
    ) -> Optional[Dict[str, Any]]:
        """
        Benchmark current model performance for analysis.

        Args:
            model: Model to benchmark (uses self.model if None)
            num_iterations: Number of benchmark iterations

        Returns:
            Benchmark results dictionary or None if benchmarking disabled
        """
        if not self.benchmarker:
            return None

        target_model = model or self.model
        if not target_model:
            self.logger_func("No model available for benchmarking")
            return None

        # Create sample input
        sample_input = torch.randn(
            1, *self.obs_shape, device=self.device, dtype=torch.float32
        )

        # Run benchmark
        self.benchmarker.benchmark_iterations = num_iterations
        result = self.benchmarker.benchmark_model(
            model=target_model,
            input_tensor=sample_input,
            name="current_model",
            model_type=self.model_type,
        )

        return {
            "mean_time_ms": result.mean_time_ms,
            "std_time_ms": result.std_time_ms,
            "memory_peak_mb": result.memory_peak_mb,
            "device": result.device,
            "num_iterations": result.num_iterations,
            "compiled": self.model_is_compiled,
        }

    # create_agent method removed as Trainer will instantiate the agent

    def handle_checkpoint_resume(
        self,
        agent: PPOAgent,
        model_dir: str,
        resume_path_override: Optional[str] = None,
    ) -> bool:
        """
        Handle resuming from checkpoint if specified or auto-detected.

        Args:
            agent: PPO agent to load checkpoint into
            model_dir: Directory to search for checkpoints

        Returns:
            bool: True if resumed from checkpoint, False otherwise
        """
        actual_resume_path = (
            resume_path_override
            if resume_path_override is not None
            else self.args.resume
        )

        if actual_resume_path == "latest" or actual_resume_path is None:
            return self._handle_latest_checkpoint_resume(agent, model_dir)
        elif actual_resume_path:
            return self._handle_specific_checkpoint_resume(agent, actual_resume_path)
        else:
            self._reset_checkpoint_state()
            return False

    def _handle_latest_checkpoint_resume(self, agent: PPOAgent, model_dir: str) -> bool:
        """Handle resuming from the latest checkpoint."""
        latest_ckpt = self._find_latest_checkpoint(model_dir)

        if latest_ckpt:
            self.checkpoint_data = agent.load_model(latest_ckpt)
            if "error" in self.checkpoint_data:
                raise RuntimeError(
                    f"Failed to load checkpoint {latest_ckpt}: {self.checkpoint_data['error']}"
                )
            self.resumed_from_checkpoint = latest_ckpt
            self.logger_func(f"Resumed from latest checkpoint: {latest_ckpt}")
            return True
        else:
            self._reset_checkpoint_state()
            self.logger_func("No checkpoint found to resume from (searched latest).")
            return False

    def _handle_specific_checkpoint_resume(
        self, agent: PPOAgent, resume_path: str
    ) -> bool:
        """Handle resuming from a specific checkpoint path."""
        self.logger_func(f"Checking if {resume_path} exists...")
        if os.path.exists(resume_path):
            self.logger_func(f"Path exists! Loading model from {resume_path}")
            self.checkpoint_data = agent.load_model(resume_path)
            if "error" in self.checkpoint_data:
                raise RuntimeError(
                    f"Failed to load checkpoint {resume_path}: {self.checkpoint_data['error']}"
                )
            self.resumed_from_checkpoint = resume_path
            self.logger_func(f"Resumed from specified checkpoint: {resume_path}")
            return True
        else:
            self.logger_func(f"Specified resume checkpoint not found: {resume_path}")
            self._reset_checkpoint_state()
            return False

    def _find_latest_checkpoint(self, model_dir: str) -> Optional[str]:
        """Find the latest checkpoint in model_dir or parent directory."""
        # Try to find latest checkpoint in the run's model_dir
        latest_ckpt = utils.find_latest_checkpoint(model_dir)

        # If not found, try the parent directory (savedir)
        if not latest_ckpt and model_dir:
            latest_ckpt = self._search_parent_directory(model_dir)

        return latest_ckpt

    def _search_parent_directory(self, model_dir: str) -> Optional[str]:
        """Search for checkpoint in parent directory and copy if found."""
        parent_dir_path = os.path.dirname(model_dir.rstrip(os.sep))
        if parent_dir_path and parent_dir_path != model_dir:
            parent_ckpt = utils.find_latest_checkpoint(parent_dir_path)
            if parent_ckpt:
                # Copy the checkpoint into the run's model_dir for consistency
                dest_ckpt = os.path.join(model_dir, os.path.basename(parent_ckpt))
                shutil.copy2(parent_ckpt, dest_ckpt)
                return dest_ckpt
        return None

    def _reset_checkpoint_state(self) -> None:
        """Reset checkpoint-related state variables."""
        self.resumed_from_checkpoint = None
        self.checkpoint_data = None

    def create_model_artifact(
        self,
        model_path: str,
        artifact_name: str,
        run_name: str,
        is_wandb_active: bool,
        artifact_type: str = "model",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
    ) -> bool:
        """
        Create and upload a W&B artifact for a model checkpoint.

        Args:
            model_path: Path to the model file to upload
            artifact_name: Name for the artifact (without run prefix)
            run_name: Current run name for artifact prefixing
            is_wandb_active: Whether WandB is currently active
            artifact_type: Type of artifact (default: "model")
            description: Optional description for the artifact
            metadata: Optional metadata dict to attach to the artifact
            aliases: Optional list of aliases (e.g., ["latest", "best"])

        Returns:
            bool: True if artifact was created successfully, False otherwise
        """
        if not (is_wandb_active and wandb.run):
            return False

        if not os.path.exists(model_path):
            self.logger_func(
                f"Warning: Model file {model_path} does not exist, skipping artifact creation."
            )
            return False

        try:
            # Create artifact with run name prefix for uniqueness
            full_artifact_name = f"{run_name}-{artifact_name}"

            # Enhance metadata with compilation information
            enhanced_metadata = metadata or {}
            if self.compilation_result:
                enhanced_metadata.update(
                    {
                        "torch_compile_enabled": getattr(
                            self.config.training, "enable_torch_compile", False
                        ),
                        "model_compiled": self.model_is_compiled,
                        "compilation_success": self.compilation_result.success,
                        "compilation_mode": getattr(
                            self.config.training, "torch_compile_mode", "default"
                        ),
                        "performance_improvement": self.compilation_result.performance_improvement,
                    }
                )

            artifact = wandb.Artifact(
                name=full_artifact_name,
                type=artifact_type,
                description=description or f"Model checkpoint from run {run_name}",
                metadata=enhanced_metadata,
            )

            # Add the model file
            artifact.add_file(model_path)

            # Log the artifact with retry logic for network failures
            self._log_artifact_with_retry(artifact, aliases, model_path)

            aliases_str = f" with aliases {aliases}" if aliases else ""
            self.logger_func(
                f"Model artifact '{full_artifact_name}' created and uploaded{aliases_str}"
            )

            return True

        except KeyboardInterrupt:
            self.logger_func(f"W&B artifact upload interrupted for {model_path}")
            return False
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            self.logger_func(f"Error creating W&B artifact for {model_path}: {e}")
            return False

    def _log_artifact_with_retry(
        self,
        artifact,
        aliases: Optional[List[str]],
        model_path: str,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        """
        Log WandB artifact with retry logic for network failures.

        Args:
            artifact: WandB artifact to log
            aliases: Optional list of aliases for the artifact
            model_path: Path to model file (for error logging)
            max_retries: Maximum number of retry attempts (default: 3)
            backoff_factor: Multiplier for exponential backoff (default: 2.0)

        Raises:
            Exception: Re-raises the last exception if all retries fail
        """
        for attempt in range(max_retries):
            try:
                wandb.log_artifact(artifact, aliases=aliases)
                return  # Success, exit retry loop
            except (ConnectionError, TimeoutError, RuntimeError) as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, re-raise the exception
                    raise e

                # Calculate backoff delay with exponential growth
                delay = backoff_factor**attempt
                self.logger_func(
                    f"WandB artifact upload attempt {attempt + 1} failed for {model_path}: {e}. "
                    f"Retrying in {delay:.1f} seconds..."
                )
                time.sleep(delay)

    def save_final_model(
        self,
        agent: PPOAgent,
        model_dir: str,
        global_timestep: int,
        total_episodes_completed: int,
        game_stats: Dict[str, int],
        run_name: str,
        is_wandb_active: bool,
    ) -> Tuple[bool, Optional[str]]:
        """
        Save the final trained model and create associated artifacts.

        Args:
            agent: PPO agent to save
            model_dir: Directory to save model in
            global_timestep: Current training timestep
            total_episodes_completed: Total episodes completed
            game_stats: Dictionary with black_wins, white_wins, draws
            run_name: Current run name
            is_wandb_active: Whether WandB is active

        Returns:
            Tuple[bool, Optional[str]]: (success, model_path)
        """
        final_model_path = os.path.join(model_dir, "final_model.pth")

        try:
            agent.save_model(
                final_model_path,
                global_timestep,
                total_episodes_completed,
            )
            self.logger_func(f"Final model saved to {final_model_path}")

            # Create W&B artifact for final model with compilation info
            final_metadata = {
                "training_timesteps": global_timestep,
                "total_episodes": total_episodes_completed,
                "black_wins": game_stats.get("black_wins", 0),
                "white_wins": game_stats.get("white_wins", 0),
                "draws": game_stats.get("draws", 0),
                "training_completed": True,
                "model_type": getattr(self.config.training, "model_type", "resnet"),
                "feature_set": getattr(self.config.env, "feature_set", "core"),
            }

            self.create_model_artifact(
                model_path=final_model_path,
                artifact_name="final-model",
                run_name=run_name,
                is_wandb_active=is_wandb_active,
                description=f"Final trained model after {global_timestep} timesteps",
                metadata=final_metadata,
                aliases=["latest", "final"],
            )

            return True, final_model_path

        except (OSError, RuntimeError) as e:
            self.logger_func(f"Error saving final model {final_model_path}: {e}")
            return False, None

    def save_checkpoint(
        self,
        agent: PPOAgent,
        model_dir: str,
        timestep: int,
        episode_count: int,
        stats: Dict[str, Any],
        run_name: str,
        is_wandb_active: bool,
        checkpoint_name_prefix: str = "checkpoint_ts",
    ) -> Tuple[bool, Optional[str]]:
        """
        Save a model checkpoint periodically.

        Args:
            agent: PPO agent to save.
            model_dir: Directory to save checkpoint in.
            timestep: Current training timestep.
            episode_count: Total episodes completed.
            stats: Dictionary with game statistics (e.g., black_wins, white_wins, draws).
            run_name: Current run name for artifact naming.
            is_wandb_active: Whether WandB is active.
            checkpoint_name_prefix: Prefix for the checkpoint filename.

        Returns:
            Tuple[bool, Optional[str]]: (success, checkpoint_path)
        """
        if timestep <= 0:
            self.logger_func("Skipping checkpoint save for timestep <= 0.")
            return False, None

        checkpoint_filename = os.path.join(
            model_dir, f"{checkpoint_name_prefix}{timestep}.pth"
        )

        # Don't save if checkpoint already exists (e.g. if called multiple times for same step)
        if os.path.exists(checkpoint_filename):
            self.logger_func(
                f"Checkpoint {checkpoint_filename} already exists. Skipping save."
            )
            return True, checkpoint_filename

        try:
            os.makedirs(model_dir, exist_ok=True)  # Ensure model_dir exists
            agent.save_model(
                checkpoint_filename,
                timestep,
                episode_count,
                stats_to_save=stats,
            )
            self.logger_func(f"Checkpoint saved to {checkpoint_filename}")

            # Create W&B artifact for this checkpoint with compilation info
            checkpoint_metadata = {
                "training_timesteps": timestep,
                "total_episodes": episode_count,
                **stats,  # Merge game stats
                "checkpoint_type": "periodic",
                "model_type": self.model_type,
                "feature_set": self.input_features,  # Assuming input_features is the feature_set
            }

            artifact_created = self.create_model_artifact(
                model_path=checkpoint_filename,
                artifact_name=f"checkpoint-ts{timestep}",
                run_name=run_name,
                is_wandb_active=is_wandb_active,
                description=f"Periodic checkpoint at timestep {timestep}",
                metadata=checkpoint_metadata,
                aliases=[f"ts-{timestep}"],  # Add a timestep specific alias
            )
            if not artifact_created and is_wandb_active:
                self.logger_func(
                    f"Warning: Failed to create WandB artifact for {checkpoint_filename}"
                )

            return True, checkpoint_filename

        except (OSError, RuntimeError) as e:
            self.logger_func(f"Error saving checkpoint {checkpoint_filename}: {e}")
            return False, None

    def save_final_checkpoint(
        self,
        agent: PPOAgent,
        model_dir: str,
        global_timestep: int,
        total_episodes_completed: int,
        game_stats: Dict[str, int],
        run_name: str,
        is_wandb_active: bool,
    ) -> Tuple[bool, Optional[str]]:
        """
        Save a final checkpoint with game statistics.

        Args:
            agent: PPO agent to save
            model_dir: Directory to save checkpoint in
            global_timestep: Current training timestep
            total_episodes_completed: Total episodes completed
            game_stats: Dictionary with black_wins, white_wins, draws
            run_name: Current run name
            is_wandb_active: Whether WandB is active

        Returns:
            Tuple[bool, Optional[str]]: (success, checkpoint_path)
        """
        if global_timestep <= 0:
            return False, None

        checkpoint_filename = os.path.join(
            model_dir, f"checkpoint_ts{global_timestep}.pth"
        )

        # Don't save if checkpoint already exists
        if os.path.exists(checkpoint_filename):
            return True, checkpoint_filename

        try:
            agent.save_model(
                checkpoint_filename,
                global_timestep,
                total_episodes_completed,
                stats_to_save=game_stats,
            )
            self.logger_func(f"Final checkpoint saved to {checkpoint_filename}")

            # Create W&B artifact for final checkpoint with compilation info
            checkpoint_metadata = {
                "training_timesteps": global_timestep,
                "total_episodes": total_episodes_completed,
                "black_wins": game_stats.get("black_wins", 0),
                "white_wins": game_stats.get("white_wins", 0),
                "draws": game_stats.get("draws", 0),
                "checkpoint_type": "final",
                "model_type": getattr(self.config.training, "model_type", "resnet"),
                "feature_set": getattr(self.config.env, "feature_set", "core"),
            }

            self.create_model_artifact(
                model_path=checkpoint_filename,
                artifact_name="final-checkpoint",
                run_name=run_name,
                is_wandb_active=is_wandb_active,
                description=f"Final checkpoint at timestep {global_timestep}",
                metadata=checkpoint_metadata,
                aliases=["latest-checkpoint"],
            )

            return True, checkpoint_filename

        except (OSError, RuntimeError) as e:
            self.logger_func(
                f"Error saving final checkpoint {checkpoint_filename}: {e}"
            )
            return False, None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration and optimization status."""
        base_info = {
            "model_type": self.model_type,
            "input_features": self.input_features,
            "tower_depth": self.tower_depth,
            "tower_width": self.tower_width,
            "se_ratio": self.se_ratio,
            "obs_shape": self.obs_shape,
            "num_planes": self.feature_spec.num_planes,
            "use_mixed_precision": self.use_mixed_precision,
            "device": str(self.device),
        }

        # Add compilation status information
        compilation_status = self.get_compilation_status()
        base_info.update(
            {
                "torch_compile": compilation_status,
                "optimization_applied": self.model_is_compiled,
                "performance_benchmarking_enabled": self.benchmarker is not None,
            }
        )

        return base_info
