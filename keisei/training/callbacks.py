"""
training/callbacks.py: Periodic task callbacks for the Shogi RL trainer.
"""

import os
import asyncio
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from keisei.evaluation.opponents.elo_registry import EloRegistry

if TYPE_CHECKING:
    from .trainer import Trainer


class Callback(ABC):
    def on_step_end(self, trainer: "Trainer"):
        # Base implementation - subclasses override this method
        pass


class AsyncCallback(ABC):
    """Base class for async callbacks."""

    async def on_step_end_async(self, trainer: "Trainer"):
        # Base implementation - subclasses override this method
        pass


class CheckpointCallback(Callback):
    def __init__(self, interval: int, model_dir: str):
        self.interval = interval
        self.model_dir = model_dir

    def on_step_end(self, trainer: "Trainer"):
        if (trainer.metrics_manager.global_timestep + 1) % self.interval == 0:
            if not trainer.agent:
                if trainer.log_both:
                    trainer.log_both(
                        "[ERROR] CheckpointCallback: Agent not initialized, cannot save checkpoint.",
                        also_to_wandb=True,
                    )
                return

            game_stats = {
                "black_wins": trainer.metrics_manager.black_wins,
                "white_wins": trainer.metrics_manager.white_wins,
                "draws": trainer.metrics_manager.draws,
            }

            # Use the consolidated save_checkpoint method from ModelManager
            success, ckpt_save_path = trainer.model_manager.save_checkpoint(
                agent=trainer.agent,
                model_dir=self.model_dir,  # model_dir is part of CheckpointCallback's state
                timestep=trainer.metrics_manager.global_timestep + 1,
                episode_count=trainer.metrics_manager.total_episodes_completed,
                stats=game_stats,
                run_name=trainer.run_name,
                is_wandb_active=trainer.is_train_wandb_active,
            )

            if success:
                if trainer.log_both and ckpt_save_path:
                    trainer.log_both(
                        f"Checkpoint saved via ModelManager to {ckpt_save_path}",
                        also_to_wandb=True,
                    )
                    if hasattr(trainer, "evaluation_manager"):
                        trainer.evaluation_manager.opponent_pool.add_checkpoint(
                            ckpt_save_path
                        )
            else:
                if trainer.log_both:
                    trainer.log_both(
                        f"[ERROR] CheckpointCallback: Failed to save checkpoint via ModelManager for timestep {trainer.metrics_manager.global_timestep + 1}.",
                        also_to_wandb=True,
                    )


class EvaluationCallback(Callback):
    def __init__(self, eval_cfg, interval: int):
        self.eval_cfg = eval_cfg
        self.interval = interval

    def on_step_end(self, trainer: "Trainer"):
        if not getattr(self.eval_cfg, "enable_periodic_evaluation", False):
            return
        if (trainer.metrics_manager.global_timestep + 1) % self.interval == 0:
            if not trainer.agent:
                if trainer.log_both:
                    trainer.log_both(
                        "[ERROR] EvaluationCallback: Agent not initialized, cannot run evaluation.",
                        also_to_wandb=True,
                    )
                return

            # Type narrowing: assert that agent is not None
            assert (
                trainer.agent is not None
            ), "Agent should be initialized at this point"

            # Ensure the model to be evaluated exists and is accessible
            # The Trainer now owns self.model directly.
            # PPOAgent also has a self.model attribute.
            # For evaluation, we typically want the agent's current model state.
            current_model = trainer.agent.model
            if not current_model:
                if trainer.log_both:
                    trainer.log_both(
                        "[ERROR] EvaluationCallback: Agent's model not found.",
                        also_to_wandb=True,
                    )
                return

            opponent_ckpt = None
            if hasattr(trainer, "evaluation_manager"):
                opponent_ckpt = trainer.evaluation_manager.opponent_pool.sample()

            # Handle initial case: no previous checkpoints available
            if opponent_ckpt is None:
                if trainer.log_both:
                    trainer.log_both(
                        "[INFO] EvaluationCallback: No previous checkpoints available. Running initial evaluation against random opponent.",
                        also_to_wandb=True,
                    )

                # Run evaluation against random opponent to bootstrap Elo system
                current_model.eval()  # Set the agent's model to eval mode
                try:
                    eval_results = trainer.evaluation_manager.evaluate_current_agent(
                        trainer.agent
                    )
                finally:
                    current_model.train()  # Set model back to train mode

                if trainer.log_both is not None:
                    trainer.log_both(
                        f"Initial evaluation completed. Results: {eval_results}",
                        also_to_wandb=True,
                        wandb_data=(
                            dict(eval_results)
                            if isinstance(eval_results, dict)
                            else {"eval_summary": str(eval_results)}
                        ),
                    )

                # Save current model as first checkpoint for future evaluations
                game_stats = {
                    "black_wins": trainer.metrics_manager.black_wins,
                    "white_wins": trainer.metrics_manager.white_wins,
                    "draws": trainer.metrics_manager.draws,
                }
                success, ckpt_path = trainer.model_manager.save_checkpoint(
                    agent=trainer.agent,
                    model_dir=trainer.session_manager.run_artifact_dir,
                    timestep=trainer.metrics_manager.global_timestep + 1,
                    episode_count=trainer.metrics_manager.total_episodes_completed,
                    stats=game_stats,
                    run_name=trainer.run_name,
                    is_wandb_active=trainer.is_train_wandb_active,
                    checkpoint_name_prefix="initial_eval_checkpoint_ts",
                )
                if success and ckpt_path and hasattr(trainer, "evaluation_manager"):
                    trainer.evaluation_manager.opponent_pool.add_checkpoint(ckpt_path)
                    if trainer.log_both:
                        trainer.log_both(
                            f"Added initial checkpoint to opponent pool: {ckpt_path}",
                            also_to_wandb=True,
                        )
                return

            if trainer.log_both is not None:
                trainer.log_both(
                    f"Starting periodic evaluation at timestep {trainer.metrics_manager.global_timestep + 1}...",
                    also_to_wandb=True,
                )

            current_model.eval()  # Set the agent's model to eval mode
            try:
                eval_results = trainer.evaluation_manager.evaluate_current_agent(
                    trainer.agent
                )
            finally:
                current_model.train()  # Set model back to train mode
            if trainer.log_both is not None:
                trainer.log_both(
                    f"Periodic evaluation finished. Results: {eval_results}",
                    also_to_wandb=True,
                    wandb_data=(
                        dict(eval_results)
                        if isinstance(eval_results, dict)
                        else {"eval_summary": str(eval_results)}
                    ),
                )

                if getattr(self.eval_cfg, "elo_registry_path", None):
                    try:
                        registry = EloRegistry(Path(self.eval_cfg.elo_registry_path))
                        snapshot = {
                            "current_id": trainer.run_name,
                            "current_rating": registry.get_rating(trainer.run_name),
                            "opponent_id": os.path.basename(str(opponent_ckpt)),
                            "opponent_rating": registry.get_rating(
                                os.path.basename(str(opponent_ckpt))
                            ),
                            "last_outcome": (
                                "win"
                                if eval_results
                                and eval_results.summary_stats.win_rate
                                > eval_results.summary_stats.loss_rate
                                else (
                                    "loss"
                                    if eval_results
                                    and eval_results.summary_stats.loss_rate
                                    > eval_results.summary_stats.win_rate
                                    else "draw"
                                )
                            ),
                            "top_ratings": sorted(
                                registry.ratings.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )[:3],
                        }
                        trainer.evaluation_elo_snapshot = snapshot
                    except (OSError, RuntimeError, ValueError, AttributeError) as e:
                        if trainer.log_both is not None:
                            trainer.log_both(
                                f"[ERROR] Failed to update Elo registry: {type(e).__name__}: {e}",
                                also_to_wandb=True,
                            )
                        trainer.evaluation_elo_snapshot = None
            # EvaluationManager always handles model mode switching; nothing else to do


class AsyncEvaluationCallback(AsyncCallback):
    """Async-native evaluation callback following Keisei patterns."""

    def __init__(self, eval_cfg, interval: int):
        self.eval_cfg = eval_cfg
        self.interval = interval

    async def on_step_end_async(self, trainer: "Trainer") -> Optional[Dict[str, float]]:
        """Async callback hook for evaluation triggers."""
        if not getattr(self.eval_cfg, "enable_periodic_evaluation", False):
            return None

        step = trainer.metrics_manager.global_timestep + 1
        if step % self.interval == 0:
            return await self._run_evaluation_async(trainer, step)
        return None

    async def _run_evaluation_async(
        self, trainer: "Trainer", step: int
    ) -> Optional[Dict[str, float]]:
        """Run evaluation in async-native way."""
        if not trainer.agent:
            if trainer.log_both:
                trainer.log_both(
                    "[ERROR] AsyncEvaluationCallback: Agent not initialized, cannot run evaluation.",
                    also_to_wandb=True,
                )
            return None

        # Type narrowing: assert that agent is not None
        assert trainer.agent is not None, "Agent should be initialized at this point"

        # Ensure the model to be evaluated exists and is accessible
        current_model = trainer.agent.model
        if not current_model:
            if trainer.log_both:
                trainer.log_both(
                    "[ERROR] AsyncEvaluationCallback: Agent's model not found.",
                    also_to_wandb=True,
                )
            return None

        opponent_ckpt = None
        if hasattr(trainer, "evaluation_manager"):
            opponent_ckpt = trainer.evaluation_manager.opponent_pool.sample()
        if opponent_ckpt is None:
            if trainer.log_both:
                trainer.log_both(
                    "[INFO] AsyncEvaluationCallback: No previous checkpoint available for Elo evaluation.",
                    also_to_wandb=False,
                )
            return None

        if trainer.log_both is not None:
            trainer.log_both(
                f"Starting async periodic evaluation at timestep {step}...",
                also_to_wandb=True,
            )

        try:
            # Use async evaluation to avoid event loop conflicts
            eval_results = (
                await trainer.evaluation_manager.evaluate_current_agent_async(
                    trainer.agent
                )
            )

            if trainer.log_both is not None:
                trainer.log_both(
                    f"Async periodic evaluation finished. Results: {eval_results}",
                    also_to_wandb=True,
                    wandb_data=(
                        dict(eval_results)
                        if isinstance(eval_results, dict)
                        else {"eval_summary": str(eval_results)}
                    ),
                )

                # Handle Elo registry updates
                if getattr(self.eval_cfg, "elo_registry_path", None):
                    try:
                        registry = EloRegistry(Path(self.eval_cfg.elo_registry_path))
                        snapshot = {
                            "current_id": trainer.run_name,
                            "current_rating": registry.get_rating(trainer.run_name),
                            "opponent_id": os.path.basename(str(opponent_ckpt)),
                            "opponent_rating": registry.get_rating(
                                os.path.basename(str(opponent_ckpt))
                            ),
                            "last_outcome": (
                                "win"
                                if eval_results
                                and eval_results.summary_stats.win_rate
                                > eval_results.summary_stats.loss_rate
                                else (
                                    "loss"
                                    if eval_results
                                    and eval_results.summary_stats.loss_rate
                                    > eval_results.summary_stats.win_rate
                                    else "draw"
                                )
                            ),
                            "top_ratings": sorted(
                                registry.ratings.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )[:3],
                        }
                        trainer.evaluation_elo_snapshot = snapshot
                    except (OSError, RuntimeError, ValueError, AttributeError) as e:
                        if trainer.log_both is not None:
                            trainer.log_both(
                                f"[ERROR] Failed to update Elo registry: {type(e).__name__}: {e}",
                                also_to_wandb=True,
                            )
                        trainer.evaluation_elo_snapshot = None

            # Return summary metrics for integration with training metrics
            if eval_results and hasattr(eval_results, "summary_stats"):
                return {
                    "evaluation/win_rate": eval_results.summary_stats.win_rate,
                    "evaluation/total_games": eval_results.summary_stats.total_games,
                    "evaluation/avg_game_length": getattr(
                        eval_results.summary_stats, "avg_game_length", 0.0
                    ),
                }
            return None

        except Exception as e:
            if trainer.log_both:
                trainer.log_both(
                    f"[ERROR] AsyncEvaluationCallback: Evaluation failed: {e}",
                    also_to_wandb=True,
                )
            return None
