"""
Main training script for Keisei Shogi RL agent.
Refactored to use the Trainer class for better modularity.
Extended with evaluation subcommand support.
"""

import argparse
import asyncio
import multiprocessing
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from keisei.config_schema import AppConfig

# Import config module and related components
from keisei.utils import load_config
from keisei.utils.unified_logger import log_error_to_stderr, log_info_to_stderr

from .trainer import Trainer
from .utils import apply_wandb_sweep_config, build_cli_overrides


def create_main_parser():
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Keisei Shogi RL Training and Evaluation System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Training command (existing functionality)
    train_parser = subparsers.add_parser("train", help="Train Shogi RL agent")
    add_training_arguments(train_parser)

    # Evaluation command (new functionality)
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained agent")
    add_evaluation_arguments(eval_parser)

    # Ladder command (continuous Elo match scheduler)
    ladder_parser = subparsers.add_parser(
        "ladder", help="Run continuous Elo ladder match scheduler"
    )
    add_ladder_arguments(ladder_parser)

    return parser


def add_training_arguments(parser):
    """Add training-specific arguments to parser."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML or JSON configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from, or 'latest' to auto-detect.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    # --- Model/feature CLI flags ---
    parser.add_argument(
        "--model", type=str, default=None, help="Model type (e.g. 'resnet')."
    )
    parser.add_argument(
        "--input_features",
        type=str,
        default=None,
        help="Feature set for observation builder.",
    )
    parser.add_argument(
        "--tower_depth", type=int, default=None, help="ResNet tower depth."
    )
    parser.add_argument(
        "--tower_width", type=int, default=None, help="ResNet tower width."
    )
    parser.add_argument(
        "--se_ratio", type=float, default=None, help="SE block squeeze ratio."
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed-precision training.",
    )
    parser.add_argument(
        "--ddp", action="store_true", help="Enable DistributedDataParallel training."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total timesteps to train for. Overrides config value.",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default=None,
        help="Directory to save models and logs. Overrides config MODEL_DIR.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values via KEY.SUBKEY=VALUE format.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=None,
        help="Update display every N steps to reduce flicker. Overrides config value.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for this run (overrides config and auto-generated name).",
    )
    # --- W&B Sweep support flags ---
    parser.add_argument(
        "--wandb-enabled",
        action="store_true",
        help="Force enable W&B logging (useful for sweeps).",
    )
    # --- Async evaluation support ---
    parser.add_argument(
        "--enable-async-evaluation",
        action="store_true",
        help="Enable async evaluation callbacks for better performance.",
    )


def add_evaluation_arguments(parser):
    """Add evaluation-specific arguments to parser."""
    parser.add_argument(
        "--agent_checkpoint",
        type=str,
        required=True,
        help="Path to agent checkpoint file to evaluate.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to evaluation configuration file.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["single_opponent", "tournament", "ladder", "benchmark", "custom"],
        help="Evaluation strategy to use.",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=None,
        help="Number of games to play during evaluation.",
    )
    parser.add_argument(
        "--opponent_type",
        type=str,
        default=None,
        help="Type of opponent: 'random', 'heuristic', 'ppo', etc.",
    )
    parser.add_argument(
        "--opponent_checkpoint",
        type=str,
        default=None,
        help="Path to opponent checkpoint (for 'ppo' opponent type).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for evaluation (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--wandb_log_eval",
        action="store_true",
        help="Enable Weights & Biases logging for evaluation.",
    )
    parser.add_argument(
        "--save_games",
        action="store_true",
        help="Save evaluation game records.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional name for this evaluation run.",
    )


def add_ladder_arguments(parser):
    """Add ladder-specific arguments to parser."""
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing model checkpoints to pit against each other.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML or JSON configuration file (used for model architecture defaults).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cpu', 'cuda', or 'cuda:N' (default: from config or 'cuda').",
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        default=None,
        help="Number of concurrent game slots (default: 6).",
    )
    parser.add_argument(
        "--num-spectated",
        type=int,
        default=None,
        help="Number of spectated (paced) game slots (default: 3).",
    )
    parser.add_argument(
        "--move-delay",
        type=float,
        default=None,
        help="Delay between moves in spectated games, in seconds (default: 1.5).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Seconds between checkpoint directory scans (default: 30).",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default=None,
        help="Path for the JSON state file consumed by the spectator dashboard.",
    )
    parser.add_argument(
        "--elo-registry",
        type=str,
        default=None,
        help="Path to Elo registry JSON file (default: <checkpoint-dir>/elo_ratings.json).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values via KEY.SUBKEY=VALUE format.",
    )


def create_agent_info_from_checkpoint(checkpoint_path: str):
    """Create AgentInfo from checkpoint path for evaluation."""
    from keisei.evaluation.core import AgentInfo

    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    return AgentInfo(
        name=checkpoint_path_obj.stem,
        checkpoint_path=str(checkpoint_path),
        metadata={"source": "cli_evaluation"},
    )


async def run_evaluation_command(args):
    """Handle standalone evaluation commands."""
    from keisei.evaluation.core_manager import EvaluationManager
    from keisei.evaluation.core import EvaluationConfig

    log_info_to_stderr("Evaluation", f"Starting evaluation of {args.agent_checkpoint}")

    # Load or create evaluation config
    if args.config:
        # Load from file and override with CLI args
        config: AppConfig = load_config(args.config, {})
        eval_config = config.evaluation
    else:
        # Create default evaluation config
        eval_config = EvaluationConfig()

    # Override config with CLI arguments
    if args.strategy is not None:
        eval_config.strategy = args.strategy
    if args.num_games is not None:
        eval_config.num_games = args.num_games
    if args.opponent_type is not None:
        eval_config.opponent_type = args.opponent_type
    if args.wandb_log_eval:
        eval_config.wandb_log_eval = args.wandb_log_eval
    if args.save_games:
        eval_config.save_games = args.save_games
    if args.output_dir:
        eval_config.save_path = args.output_dir

    # Configure strategy-specific parameters
    if eval_config.strategy == "single_opponent":
        eval_config.configure_for_single_opponent(
            opponent_name=args.opponent_type or eval_config.opponent_type,
            opponent_path=args.opponent_checkpoint,
        )

    # Generate run name if not provided
    run_name = args.run_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create evaluation manager
    manager = EvaluationManager(
        config=eval_config,
        run_name=run_name,
        pool_size=eval_config.previous_model_pool_size,
        elo_registry_path=eval_config.elo_registry_path,
    )

    # Setup device and runtime context (minimal for CLI evaluation)
    manager.setup(
        device=args.device,
        policy_mapper=None,  # Will be created by evaluator if needed
        model_dir=str(Path(args.agent_checkpoint).parent),
        wandb_active=args.wandb_log_eval,
    )

    try:
        # Create agent info and run evaluation
        agent_info = create_agent_info_from_checkpoint(args.agent_checkpoint)

        log_info_to_stderr(
            "Evaluation",
            f"Running {eval_config.strategy} evaluation with {eval_config.num_games} games",
        )

        # Use async evaluation for better performance
        result = await manager.evaluate_checkpoint_async(
            args.agent_checkpoint, args.opponent_checkpoint
        )

        # Display results
        log_info_to_stderr("Evaluation", f"Evaluation complete!")
        log_info_to_stderr("Evaluation", f"Results: {result.summary_stats}")

        if hasattr(result.summary_stats, "win_rate"):
            log_info_to_stderr(
                "Evaluation", f"Win Rate: {result.summary_stats.win_rate:.2%}"
            )
        if hasattr(result.summary_stats, "total_games"):
            log_info_to_stderr(
                "Evaluation", f"Total Games: {result.summary_stats.total_games}"
            )

        # Save results if requested
        if eval_config.save_games and eval_config.save_path:
            output_path = Path(eval_config.save_path)
            output_path.mkdir(parents=True, exist_ok=True)
            result_file = output_path / f"{run_name}_results.json"

            # Save evaluation results (simplified)
            import json

            results_data = {
                "agent_checkpoint": args.agent_checkpoint,
                "evaluation_config": eval_config.model_dump(),
                "results": (
                    result.model_dump()
                    if hasattr(result, "model_dump")
                    else str(result)
                ),
                "timestamp": datetime.now().isoformat(),
            }

            with open(result_file, "w") as f:
                json.dump(results_data, f, indent=2, default=str)

            log_info_to_stderr("Evaluation", f"Results saved to {result_file}")

        return result

    except Exception as e:
        log_error_to_stderr("Evaluation", f"Evaluation failed: {e}")
        raise


async def run_ladder_command(args):
    """Handle the ladder subcommand: run the ContinuousMatchScheduler."""
    from keisei.evaluation.scheduler import ContinuousMatchScheduler, SchedulerConfig

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        log_error_to_stderr("Ladder", f"Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Load config for model architecture defaults
    cli_overrides = {}
    for override in args.override:
        if "=" not in override:
            log_error_to_stderr(
                "Ladder", f"Malformed override '{override}': expected KEY=VALUE format"
            )
            sys.exit(1)
        k, v = override.split("=", 1)
        cli_overrides[k] = v

    config: Optional[AppConfig] = None
    if args.config:
        config = load_config(args.config, cli_overrides)
    elif cli_overrides:
        config = load_config(None, cli_overrides)

    # Build SchedulerConfig, pulling model architecture from AppConfig if available
    scheduler_kwargs: dict = {
        "checkpoint_dir": checkpoint_dir,
        "elo_registry_path": Path(
            args.elo_registry
            if args.elo_registry
            else str(checkpoint_dir / "elo_ratings.json")
        ),
    }

    # Device: CLI flag > config > default
    if args.device is not None:
        scheduler_kwargs["device"] = args.device
    elif config is not None:
        scheduler_kwargs["device"] = config.env.device

    # Scheduler-specific CLI overrides
    if args.num_concurrent is not None:
        scheduler_kwargs["num_concurrent"] = args.num_concurrent
    if args.num_spectated is not None:
        scheduler_kwargs["num_spectated"] = args.num_spectated
    if args.move_delay is not None:
        scheduler_kwargs["move_delay"] = args.move_delay
    if args.poll_interval is not None:
        scheduler_kwargs["poll_interval"] = args.poll_interval
    if args.state_path is not None:
        scheduler_kwargs["state_path"] = Path(args.state_path)

    # Pull model architecture from AppConfig training section
    if config is not None:
        t = config.training
        scheduler_kwargs.setdefault("input_channels", config.env.input_channels)
        scheduler_kwargs.setdefault("input_features", t.input_features)
        scheduler_kwargs.setdefault("model_type", t.model_type)
        scheduler_kwargs.setdefault("tower_depth", t.tower_depth)
        scheduler_kwargs.setdefault("tower_width", t.tower_width)
        scheduler_kwargs.setdefault("se_ratio", t.se_ratio)

    from pydantic import ValidationError

    try:
        scheduler_config = SchedulerConfig(**scheduler_kwargs)
    except (ValidationError, ValueError) as e:
        log_error_to_stderr("Ladder", f"Invalid configuration: {e}")
        sys.exit(1)
    scheduler = ContinuousMatchScheduler(scheduler_config)

    log_info_to_stderr("Ladder", f"Checkpoint dir: {checkpoint_dir}")
    log_info_to_stderr("Ladder", f"Device: {scheduler_config.device}")
    log_info_to_stderr(
        "Ladder",
        f"Slots: {scheduler_config.num_concurrent} concurrent, "
        f"{scheduler_config.num_spectated} spectated",
    )
    log_info_to_stderr("Ladder", "Starting scheduler (Ctrl+C to stop)...")

    try:
        await scheduler.run()
    except (asyncio.CancelledError, KeyboardInterrupt):
        log_info_to_stderr("Ladder", "Scheduler stopped.")


def run_training_command(args):
    """Handle training commands (existing functionality)."""
    # Get W&B sweep overrides if running in a sweep
    sweep_overrides = apply_wandb_sweep_config()

    # Build CLI overrides dict (dot notation)
    cli_overrides = {}
    for override in args.override:
        if "=" not in override:
            log_error_to_stderr(
                "Training", f"Malformed override '{override}': expected KEY=VALUE format"
            )
            sys.exit(1)
        k, v = override.split("=", 1)
        cli_overrides[k] = v

    # Add direct CLI args as overrides using shared utility
    direct_cli_overrides = build_cli_overrides(args)
    cli_overrides.update(direct_cli_overrides)
    # Do NOT generate run_name here; let Trainer handle it for correct CLI/config/auto priority

    # Merge sweep overrides with CLI overrides (CLI takes precedence)
    final_overrides = {**sweep_overrides, **cli_overrides}

    # Load config (YAML/JSON + CLI overrides + sweep overrides)
    config: AppConfig = load_config(args.config, final_overrides)

    # Initialize and run the trainer (Trainer will determine run_name)
    trainer = Trainer(config=config, args=args)

    # Enable async evaluation if requested
    if getattr(args, "enable_async_evaluation", False):
        trainer.callback_manager.use_async_evaluation()
        log_info_to_stderr("Training", "Async evaluation callbacks enabled")

    trainer.run_training_loop()


async def main():
    """Extended main function with evaluation support."""
    parser = create_main_parser()

    # If no arguments provided, default to training with help
    if len(sys.argv) == 1:
        # Show help and set default command to train
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.command == "evaluate":
        # Pure evaluation mode
        await run_evaluation_command(args)
    elif args.command == "train":
        # Training mode (with optional evaluation)
        run_training_command(args)
    elif args.command == "ladder":
        # Continuous Elo ladder match scheduler
        await run_ladder_command(args)
    else:
        # No command specified, show help
        parser.print_help()
        sys.exit(1)


def _run_async_entrypoint(async_fn):
    """Run an async entrypoint and close the coroutine if startup fails early."""
    coro = async_fn()
    try:
        return asyncio.run(coro)
    except BaseException:
        close = getattr(coro, "close", None)
        if callable(close):
            close()
        raise


def main_sync():
    """Synchronous entry point that handles async main properly."""
    # Fix B6: Add multiprocessing.freeze_support() for Windows compatibility
    multiprocessing.freeze_support()

    # Set multiprocessing start method for safety, especially with CUDA
    try:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError as e:
        log_error_to_stderr(
            "Train",
            f"Could not set multiprocessing start method to 'spawn': {e}. Using default: {multiprocessing.get_start_method(allow_none=True)}",
        )
    except Exception as e:
        log_error_to_stderr("Train", f"Error setting multiprocessing start_method: {e}")

    # Run the async main function
    try:
        _run_async_entrypoint(main)
    except KeyboardInterrupt:
        log_info_to_stderr("Main", "Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error_to_stderr("Main", f"Process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_sync()
