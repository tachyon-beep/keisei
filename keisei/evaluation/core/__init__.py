"""
Core evaluation system components.

This module contains the foundational classes and abstractions for the
evaluation system, including contexts, results, and base evaluator classes.
"""

from typing import Optional

from .base_evaluator import (
    BaseEvaluator,
    EvaluatorFactory,
    create_agent_info,
    evaluate_agent,
)
from keisei.config_schema import EvaluationConfig, EvaluationStrategy
from .evaluation_context import AgentInfo, EvaluationContext, OpponentInfo
from .evaluation_result import (
    EvaluationResult,
    GameResult,
    SummaryStats,
    create_game_result,
)
from .model_manager import ModelWeightManager
from .parallel_executor import (
    BatchGameExecutor,
    ParallelGameExecutor,
    ParallelGameTask,
    create_parallel_game_tasks,
)


# Config factory function
def create_evaluation_config(
    strategy: str = "single_opponent",
    num_games: int = 10,
    max_concurrent_games: int = 4,
    timeout_per_game: float = 60.0,
    randomize_positions: bool = True,
    random_seed: Optional[int] = None,
    save_games: bool = True,
    save_path: Optional[str] = None,
    log_level: str = "INFO",
    wandb_logging: bool = False,
    update_elo: bool = True,
    enable_in_memory_evaluation: bool = True,
    model_weight_cache_size: int = 5,
    enable_parallel_execution: bool = True,
    process_restart_threshold: int = 100,
    temp_agent_device: str = "cpu",
    clear_cache_after_evaluation: bool = True,
    opponent_name: str = "random",
    **kwargs,
) -> EvaluationConfig:
    """
    Create an EvaluationConfig instance with the specified parameters.

    This factory function provides a convenient way to create evaluation
    configurations for the unified evaluation system.
    """
    return EvaluationConfig(
        strategy=strategy,
        num_games=num_games,
        max_concurrent_games=max_concurrent_games,
        timeout_per_game=timeout_per_game,
        randomize_positions=randomize_positions,
        random_seed=random_seed,
        save_games=save_games,
        save_path=save_path,
        log_level=log_level,
        wandb_log_eval=wandb_logging,
        update_elo=update_elo,
        enable_in_memory_evaluation=enable_in_memory_evaluation,
        model_weight_cache_size=model_weight_cache_size,
        enable_parallel_execution=enable_parallel_execution,
        process_restart_threshold=process_restart_threshold,
        temp_agent_device=temp_agent_device,
        clear_cache_after_evaluation=clear_cache_after_evaluation,
        opponent_type=opponent_name,
        strategy_params=kwargs.get("strategy_params", {}),
        **{k: v for k, v in kwargs.items() if k != "strategy_params"},
    )


__all__ = [
    # Base classes
    "BaseEvaluator",
    "EvaluatorFactory",
    "evaluate_agent",
    "create_agent_info",
    # Configuration
    "EvaluationConfig",
    "EvaluationStrategy",
    "create_evaluation_config",
    # Context and metadata
    "EvaluationContext",
    "AgentInfo",
    "OpponentInfo",
    # Results
    "EvaluationResult",
    "GameResult",
    "SummaryStats",
    "create_game_result",
    # Model management
    "ModelWeightManager",
    # Parallel execution
    "ParallelGameExecutor",
    "BatchGameExecutor",
    "ParallelGameTask",
    "create_parallel_game_tasks",
]

# Enhanced features (optional)
try:
    from .background_tournament import (
        BackgroundTournamentManager,
        TournamentProgress,
        TournamentStatus,
    )

    __all__.extend(
        ["BackgroundTournamentManager", "TournamentProgress", "TournamentStatus"]
    )
except ImportError:
    # Background tournament features not available
    pass
