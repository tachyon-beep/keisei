"""
agent_loading.py: Utilities for loading PPO agents and initializing opponents.
"""

import os
from typing import Any, Optional

from keisei.utils.opponents import (
    SimpleHeuristicOpponent,
    SimpleRandomOpponent,
)
from keisei.utils.unified_logger import log_error_to_stderr, log_info_to_stderr


def _build_evaluation_config(
    device_str: str,
    policy_mapper,
    input_channels: int,
    input_features: Optional[str] = "core46",
    model_type: str = "resnet",
    tower_depth: int = 9,
    tower_width: int = 256,
    se_ratio: Optional[float] = 0.25,
) -> Any:
    """Build a minimal AppConfig suitable for loading an evaluation agent."""
    from keisei.config_schema import (  # pylint: disable=import-outside-toplevel
        AppConfig,
        DemoConfig,
        DisplayConfig,
        EnvConfig,
        EvaluationConfig,
        LoggingConfig,
        ParallelConfig,
        TrainingConfig,
        WandBConfig,
    )

    return AppConfig(
        parallel=ParallelConfig(
            enabled=False,
            num_workers=1,
            batch_size=32,
            sync_interval=100,
            compression_enabled=True,
            timeout_seconds=10.0,
            max_queue_size=1000,
            worker_seed_offset=1000,
        ),
        env=EnvConfig(
            device=device_str,
            input_channels=input_channels,
            num_actions_total=policy_mapper.get_total_actions(),
            seed=42,
            max_moves_per_game=500,
        ),
        training=TrainingConfig(
            total_timesteps=1,
            steps_per_epoch=1,
            ppo_epochs=1,
            minibatch_size=2,
            learning_rate=1e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            input_features=input_features or "core46",
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=True,
            tower_depth=tower_depth,
            tower_width=tower_width,
            se_ratio=se_ratio,
            model_type=model_type,
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=10000,
            evaluation_interval_timesteps=50000,
            weight_decay=0.0,
            normalize_advantages=True,
            enable_value_clipping=False,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            enable_periodic_evaluation=False,
            evaluation_interval_timesteps=50000,
            strategy="single_opponent",
            num_games=1,
            max_concurrent_games=4,
            timeout_per_game=None,
            opponent_type="random",
            max_moves_per_game=500,
            randomize_positions=True,
            random_seed=None,
            save_games=True,
            save_path=None,
            log_file_path_eval="/tmp/eval.log",
            log_level="INFO",
            wandb_log_eval=False,
            update_elo=True,
            elo_registry_path="elo_ratings.json",
            agent_id=None,
            opponent_id=None,
            previous_model_pool_size=5,
            enable_in_memory_evaluation=True,
            model_weight_cache_size=5,
            enable_parallel_execution=True,
            process_restart_threshold=100,
            temp_agent_device="cpu",
            clear_cache_after_evaluation=True,
        ),
        logging=LoggingConfig(
            log_file="/tmp/eval.log", model_dir="/tmp/", run_name="eval-run"
        ),
        wandb=WandBConfig(
            enabled=False,
            project="eval",
            entity=None,
            run_name_prefix="eval-run",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
            log_model_artifact=False,
        ),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0),
        display=DisplayConfig(
            enable_board_display=True,
            enable_trend_visualization=True,
            enable_elo_ratings=True,
            enable_enhanced_layout=True,
            display_moves=False,
            turn_tick=0.5,
            board_unicode_pieces=True,
            board_cell_width=5,
            board_cell_height=3,
            board_highlight_last_move=True,
            sparkline_width=15,
            trend_history_length=100,
            elo_initial_rating=1500.0,
            elo_k_factor=32.0,
            dashboard_height_ratio=2,
            progress_bar_height=4,
            show_text_moves=True,
            move_list_length=10,
            moves_latest_top=True,
            moves_flash_ms=500,
            show_moves_trend=True,
            show_completion_rate=True,
            show_enhanced_win_rates=True,
            show_turns_trend=True,
            metrics_window_size=100,
            trend_smoothing_factor=0.1,
            metrics_panel_height=6,
            enable_trendlines=True,
            log_layer_keyword_filters=["stem", "policy_head", "value_head"],
        ),
    )


def _load_model_from_checkpoint(
    checkpoint_path: str,
    device_str: str,
    policy_mapper,
    input_channels: int,
    model_type: str = "resnet",
    tower_depth: int = 9,
    tower_width: int = 256,
    se_ratio: Optional[float] = 0.25,
) -> Any:
    """Create a model using model_factory and return it with the device."""
    import torch  # pylint: disable=import-outside-toplevel

    from keisei.training.models import (  # pylint: disable=import-outside-toplevel
        model_factory,
    )

    device = torch.device(device_str)
    num_actions = policy_mapper.get_total_actions()
    obs_shape = (input_channels, 9, 9)

    temp_model = model_factory(
        model_type=model_type,
        obs_shape=obs_shape,
        num_actions=num_actions,
        tower_depth=tower_depth,
        tower_width=tower_width,
        se_ratio=se_ratio,
    ).to(device)
    return temp_model, device


def _create_ppo_agent(
    model: Any,
    config: Any,
    device: Any,
    checkpoint_path: str,
) -> Any:
    """Instantiate a PPOAgent, load checkpoint, and set to eval mode."""
    from keisei.core.ppo_agent import (  # pylint: disable=import-outside-toplevel
        PPOAgent,
    )

    agent = PPOAgent(
        model=model, config=config, device=device, name="EvaluationAgent"
    )
    agent.load_model(checkpoint_path)
    agent.model.eval()
    return agent


def load_evaluation_agent(
    checkpoint_path: str,
    device_str: str,
    policy_mapper,
    input_channels: int,
    input_features: Optional[str] = "core46",
    model_type: str = "resnet",
    tower_depth: int = 9,
    tower_width: int = 256,
    se_ratio: Optional[float] = 0.25,
) -> Any:
    if not os.path.isfile(checkpoint_path):
        log_error_to_stderr(
            "AgentLoading", f"Checkpoint file {checkpoint_path} not found"
        )
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")

    config = _build_evaluation_config(
        device_str, policy_mapper, input_channels, input_features,
        model_type=model_type,
        tower_depth=tower_depth,
        tower_width=tower_width,
        se_ratio=se_ratio,
    )

    temp_model, device = _load_model_from_checkpoint(
        checkpoint_path, device_str, policy_mapper, input_channels,
        model_type=model_type,
        tower_depth=tower_depth,
        tower_width=tower_width,
        se_ratio=se_ratio,
    )

    agent = _create_ppo_agent(temp_model, config, device, checkpoint_path)

    log_info_to_stderr(
        "AgentLoading",
        f"Loaded agent from {checkpoint_path} on device {device_str} for evaluation",
    )
    return agent


def initialize_opponent(
    opponent_type: str,
    opponent_path: Optional[str],
    device_str: str,
    policy_mapper,
    input_channels: int,
) -> Any:
    if opponent_type == "random":
        return SimpleRandomOpponent()
    if opponent_type == "heuristic":
        return SimpleHeuristicOpponent()
    if opponent_type == "ppo":
        if not opponent_path:
            raise ValueError("Opponent path must be provided for PPO opponent type.")
        return load_evaluation_agent(
            opponent_path, device_str, policy_mapper, input_channels
        )

    raise ValueError(f"Unknown opponent type: {opponent_type}")


__all__ = [
    "load_evaluation_agent",
    "initialize_opponent",
]
