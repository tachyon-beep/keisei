"""
training/utils.py: Helper functions for setup and configuration in the Shogi RL trainer.
"""

import glob
import json
import os
import pickle
import random
import sys
from typing import Optional

import numpy as np
import torch

import wandb
from keisei.config_schema import AppConfig
from keisei.utils.unified_logger import log_error_to_stderr, log_info_to_stderr


def _validate_checkpoint(checkpoint_path: str) -> bool:
    """Validate checkpoint file integrity by attempting to load it.

    Args:
        checkpoint_path: Path to checkpoint file to validate

    Returns:
        True if checkpoint loads successfully, False otherwise
    """
    try:
        # Attempt minimal load to check file integrity
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        return True
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as e:
        log_error_to_stderr(
            "TrainingUtils", f"Corrupted checkpoint {checkpoint_path}: {e}"
        )
        return False


def find_latest_checkpoint(model_dir_path: str) -> Optional[str]:
    try:
        checkpoints = glob.glob(os.path.join(model_dir_path, "*.pth"))
        if not checkpoints:
            checkpoints = glob.glob(os.path.join(model_dir_path, "*.pt"))
        if not checkpoints:
            return None

        # Sort checkpoints by modification time (newest first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)

        # Find the first valid (non-corrupted) checkpoint
        for checkpoint_path in checkpoints:
            if _validate_checkpoint(checkpoint_path):
                return checkpoint_path

        # If we get here, all checkpoints are corrupted
        log_error_to_stderr(
            "TrainingUtils",
            "All checkpoint files in directory are corrupted or unreadable",
        )
        return None

    except OSError as e:
        log_error_to_stderr("TrainingUtils", f"Error in find_latest_checkpoint: {e}")
        return None


def serialize_config(config: AppConfig) -> str:
    """Serialize AppConfig to JSON string using Pydantic's built-in serialization.

    Args:
        config: AppConfig instance to serialize

    Returns:
        JSON string representation of the configuration
    """
    return config.model_dump_json(indent=4)


def setup_directories(config, run_name):
    model_dir = config.logging.model_dir
    log_file = config.logging.log_file
    run_artifact_dir = os.path.join(model_dir, run_name)
    model_dir_path = run_artifact_dir
    log_file_path = os.path.join(run_artifact_dir, os.path.basename(log_file))
    eval_log_file_path = os.path.join(run_artifact_dir, "rich_periodic_eval_log.txt")
    os.makedirs(run_artifact_dir, exist_ok=True)
    return {
        "run_artifact_dir": run_artifact_dir,
        "model_dir": model_dir_path,
        "log_file_path": log_file_path,
        "eval_log_file_path": eval_log_file_path,
    }


def setup_seeding(config):
    seed = config.env.seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def setup_wandb(config, run_name, run_artifact_dir):
    wandb_cfg = config.wandb
    is_active = wandb_cfg.enabled
    if is_active:
        try:
            # Debug wandb import state
            import wandb as local_wandb

            log_info_to_stderr(
                "TrainingUtils",
                f"wandb module type: {type(local_wandb)}, has init: {hasattr(local_wandb, 'init')}",
            )

            config_dict_for_wandb = (
                json.loads(serialize_config(config)) if serialize_config(config) else {}
            )
            local_wandb.init(
                project=wandb_cfg.project,
                entity=wandb_cfg.entity,
                name=run_name,
                config=config_dict_for_wandb,
                mode="online" if wandb_cfg.enabled else "disabled",
                dir=run_artifact_dir,
                resume="allow",
                id=run_name,
            )
        except (TypeError, ValueError, OSError, AttributeError) as e:
            # Add specific debugging for AttributeError
            if isinstance(e, AttributeError):
                import traceback

                error_trace = traceback.format_exc()
                log_error_to_stderr(
                    "TrainingUtils",
                    f"AttributeError in W&B init: {e}. Traceback:\n{error_trace}",
                )
            else:
                log_error_to_stderr(
                    "TrainingUtils",
                    f"Error initializing W&B: {e}. W&B logging disabled.",
                )
            is_active = False
    if not is_active:
        log_info_to_stderr(
            "TrainingUtils",
            "Weights & Biases logging is disabled or failed to initialize.",
        )
    return is_active


def apply_wandb_sweep_config():
    """Apply W&B sweep configuration to override parameters.

    Returns:
        Dict[str, Any]: Dictionary of configuration overrides extracted from sweep parameters.
                       Empty dict if no W&B sweep is active.
    """
    try:
        if wandb.run is None:
            return {}
        sweep_config = wandb.config
    except (AttributeError, ImportError):
        # WandB not properly initialized or not available
        return {}
    log_info_to_stderr(
        "TrainingUtils", f"Running W&B sweep with config: {dict(sweep_config)}"
    )

    # Map W&B sweep parameters to config paths
    sweep_param_mapping = {
        "learning_rate": "training.learning_rate",
        "gamma": "training.gamma",
        "clip_epsilon": "training.clip_epsilon",
        "ppo_epochs": "training.ppo_epochs",
        "minibatch_size": "training.minibatch_size",
        "value_loss_coeff": "training.value_loss_coeff",
        "entropy_coef": "training.entropy_coef",
        "tower_depth": "training.tower_depth",
        "tower_width": "training.tower_width",
        "se_ratio": "training.se_ratio",
        "steps_per_epoch": "training.steps_per_epoch",
        "gradient_clip_max_norm": "training.gradient_clip_max_norm",
        "lambda_gae": "training.lambda_gae",
    }

    # Apply sweep parameters as overrides
    sweep_overrides = {"wandb.enabled": True}  # Force enable W&B for sweeps
    for sweep_key, config_path in sweep_param_mapping.items():
        if hasattr(sweep_config, sweep_key):
            sweep_overrides[config_path] = getattr(sweep_config, sweep_key)

    return sweep_overrides


def build_cli_overrides(args) -> dict:
    """
    Build configuration overrides from command line arguments.

    This function centralizes the logic for converting CLI arguments
    to configuration overrides, eliminating duplication between training scripts.

    Args:
        args: Parsed command line arguments from argparse

    Returns:
        dict: Configuration overrides in dot notation format
    """
    cli_overrides = {}

    # Core overrides supported by both training scripts
    if hasattr(args, "seed") and args.seed is not None:
        cli_overrides["env.seed"] = args.seed
    if hasattr(args, "device") and args.device is not None:
        cli_overrides["env.device"] = args.device
    if hasattr(args, "total_timesteps") and args.total_timesteps is not None:
        cli_overrides["training.total_timesteps"] = args.total_timesteps

    # Extended overrides only available in main training script
    if hasattr(args, "savedir") and args.savedir is not None:
        cli_overrides["logging.model_dir"] = args.savedir
    if hasattr(args, "render_every") and args.render_every is not None:
        cli_overrides["training.render_every_steps"] = args.render_every
    if hasattr(args, "wandb_enabled") and args.wandb_enabled:
        cli_overrides["wandb.enabled"] = True

    return cli_overrides
