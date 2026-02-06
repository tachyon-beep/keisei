"""
Pydantic configuration schema for Keisei DRL Shogi Client.
Defines all configuration sections and their defaults.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Evaluation strategy constants - matches EvaluationStrategy enum values
VALID_EVALUATION_STRATEGIES = [
    "single_opponent",
    "tournament",
    "ladder",
    "benchmark",
    "custom",
]


# Strategy constants for factory registration (backward compatibility)
class EvaluationStrategy:
    SINGLE_OPPONENT = "single_opponent"
    TOURNAMENT = "tournament"
    LADDER = "ladder"
    BENCHMARK = "benchmark"
    CUSTOM = "custom"


class EnvConfig(BaseModel):
    device: str = Field("cpu", description="Device to use: 'cpu' or 'cuda'.")
    input_channels: int = Field(
        46, description="Number of input channels for the neural network."
    )
    num_actions_total: int = Field(
        13527, description="Total number of possible actions."
    )
    seed: int = Field(42, description="Random seed for reproducibility.")
    max_moves_per_game: int = Field(
        500, description="Maximum number of moves per game before declaring a draw."
    )


class TrainingConfig(BaseModel):
    total_timesteps: int = Field(
        500_000, description="Total number of environment steps for training."
    )
    steps_per_epoch: int = Field(2048, description="Steps per PPO buffer/epoch.")
    ppo_epochs: int = Field(10, description="Number of PPO update epochs per buffer.")
    minibatch_size: int = Field(
        64,
        gt=1,
        description="Minibatch size for PPO updates. Must be > 1 to avoid issues with std calculation.",
    )
    learning_rate: float = Field(3e-4, description="Learning rate for optimizer.")
    gamma: float = Field(0.99, description="Discount factor.")
    clip_epsilon: float = Field(0.2, description="PPO clip epsilon.")
    value_loss_coeff: float = Field(0.5, description="Value loss coefficient.")
    entropy_coef: float = Field(0.01, description="Entropy regularization coefficient.")
    render_every_steps: int = Field(
        1,
        description="Update expensive display elements (metrics, logs) every N steps to reduce flicker.",
    )
    refresh_per_second: int = Field(4, description="Rich Live refresh rate per second.")
    enable_spinner: bool = Field(
        True, description="Enable spinner column in progress bar (looks cool!)."
    )
    # --- Model/feature config additions ---
    input_features: str = Field(
        "core46",
        description="Feature set for observation builder (e.g. 'core46', 'core46+all').",
    )
    tower_depth: int = Field(
        9, description="Number of residual blocks in ResNet tower."
    )
    tower_width: int = Field(256, description="Width (channels) of ResNet tower.")
    se_ratio: float = Field(
        0.25, description="SE block squeeze ratio (0 disables SE blocks)."
    )
    model_type: str = Field("resnet", description="Model type to use (e.g. 'resnet').")
    mixed_precision: bool = Field(False, description="Enable mixed-precision training.")
    ddp: bool = Field(False, description="Enable DistributedDataParallel training.")
    gradient_clip_max_norm: float = Field(
        0.5, description="Maximum norm for gradient clipping."
    )
    lambda_gae: float = Field(
        0.95, description="Lambda for Generalized Advantage Estimation (GAE)."
    )
    checkpoint_interval_timesteps: int = Field(
        10000, description="Save a model checkpoint every N timesteps."
    )
    evaluation_interval_timesteps: int = Field(
        50000, description="Run evaluation every N timesteps."
    )
    weight_decay: float = Field(
        0.0, description="Weight decay (L2 regularization) for optimizer."
    )
    # PPO-specific normalization options
    normalize_advantages: bool = Field(
        True,
        description="Enable advantage normalization in PPO training for improved stability.",
    )

    # Value function clipping configuration
    enable_value_clipping: bool = Field(
        False,
        description="Enable value function loss clipping to stabilize training.",
    )

    # Learning Rate Scheduling Configuration
    lr_schedule_type: Optional[str] = Field(
        None,
        description="Type of learning rate scheduler: 'linear', 'cosine', 'exponential', 'step', or None to disable",
    )
    lr_schedule_kwargs: Optional[dict] = Field(
        None, description="Additional keyword arguments for the learning rate scheduler"
    )
    lr_schedule_step_on: str = Field(
        "epoch",
        description="When to step the scheduler: 'epoch' (per PPO epoch) or 'update' (per minibatch update)",
    )

    # torch.compile Configuration - Neural Network Optimization
    enable_torch_compile: bool = Field(
        True,
        description="Enable torch.compile for neural network optimization (10-30% speedup).",
    )
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        "default",
        description="torch.compile mode: 'default' (balanced), 'reduce-overhead' (faster compilation), 'max-autotune' (maximum optimization).",
    )
    torch_compile_dynamic: Optional[bool] = Field(
        None,
        description="Enable dynamic shape tracing for variable input sizes (None for auto-detection).",
    )
    torch_compile_fullgraph: bool = Field(
        False,
        description="Force full graph compilation (may fail on complex models but provides maximum optimization).",
    )
    torch_compile_backend: Optional[str] = Field(
        None,
        description="Compilation backend ('inductor', 'aot_eager', 'cudagraphs'). None for auto-selection.",
    )
    enable_compilation_fallback: bool = Field(
        True,
        description="Automatically fallback to non-compiled model if compilation fails.",
    )
    validate_compiled_output: bool = Field(
        True,
        description="Validate numerical equivalence between compiled and non-compiled models during initialization.",
    )
    compilation_validation_tolerance: float = Field(
        1e-5,
        description="Numerical tolerance for compiled model validation (absolute difference).",
    )
    compilation_warmup_steps: int = Field(
        5, description="Number of warmup forward passes for torch.compile optimization."
    )
    enable_compilation_benchmarking: bool = Field(
        True,
        description="Enable performance benchmarking for compiled vs non-compiled models.",
    )

    @field_validator("learning_rate")
    # pylint: disable=no-self-argument
    def lr_positive(cls, v):
        if v <= 0:
            raise ValueError("learning_rate must be positive")
        return v

    @field_validator("lr_schedule_type")
    def validate_lr_schedule_type(cls, v):  # pylint: disable=no-self-argument
        if v is not None and v not in ["linear", "cosine", "exponential", "step"]:
            raise ValueError(
                "lr_schedule_type must be one of: 'linear', 'cosine', 'exponential', 'step', or None"
            )
        return v

    @field_validator("lr_schedule_step_on")
    def validate_lr_schedule_step_on(cls, v):  # pylint: disable=no-self-argument
        if v not in ["epoch", "update"]:
            raise ValueError("lr_schedule_step_on must be 'epoch' or 'update'")
        return v

    @field_validator("torch_compile_mode")
    def validate_torch_compile_mode(cls, v):  # pylint: disable=no-self-argument
        if v not in ["default", "reduce-overhead", "max-autotune"]:
            raise ValueError(
                "torch_compile_mode must be one of: 'default', 'reduce-overhead', 'max-autotune'"
            )
        return v

    @field_validator("compilation_validation_tolerance")
    def validate_compilation_tolerance(cls, v):  # pylint: disable=no-self-argument
        if v <= 0:
            raise ValueError("compilation_validation_tolerance must be positive")
        return v

    @field_validator("compilation_warmup_steps")
    def validate_warmup_steps(cls, v):  # pylint: disable=no-self-argument
        if v < 0:
            raise ValueError("compilation_warmup_steps must be non-negative")
        return v


class EvaluationConfig(BaseModel):
    # Core periodic evaluation settings
    enable_periodic_evaluation: bool = Field(
        True, description="Enable periodic evaluation during training."
    )
    evaluation_interval_timesteps: int = Field(
        50000, description="Run evaluation every N timesteps."
    )

    # Strategy and game parameters
    strategy: Literal[
        "single_opponent", "tournament", "ladder", "benchmark", "custom"
    ] = Field(
        "single_opponent",
        description="Evaluation strategy: 'single_opponent', 'tournament', 'ladder', 'benchmark'",
    )
    num_games: int = Field(20, description="Number of games to play during evaluation.")
    max_concurrent_games: int = Field(
        4, description="Maximum number of concurrent games for parallel execution."
    )
    timeout_per_game: Optional[float] = Field(
        None, description="Timeout per game in seconds (None for no timeout)."
    )

    # Strategy-specific parameters
    strategy_params: dict = Field(
        default_factory=dict,
        description="Strategy-specific parameters for advanced configuration.",
    )

    # Game configuration
    opponent_type: str = Field(
        "random", description="Type of opponent: 'random', 'heuristic', etc."
    )
    max_moves_per_game: int = Field(
        500, description="Maximum number of moves per evaluation game."
    )
    randomize_positions: bool = Field(
        True, description="Randomize starting positions for evaluation games."
    )
    random_seed: Optional[int] = Field(
        None, description="Random seed for evaluation (None for random)."
    )

    # Output and logging
    save_games: bool = Field(True, description="Save evaluation game records.")
    save_path: Optional[str] = Field(
        None, description="Path to save evaluation results (None for auto-generated)."
    )
    log_file_path_eval: str = Field(
        "eval_log.txt", description="Path for the evaluation log file."
    )
    log_level: str = Field(
        "INFO", description="Logging level for evaluation: DEBUG, INFO, WARNING, ERROR"
    )
    wandb_log_eval: bool = Field(
        False, description="Enable Weights & Biases logging for evaluation."
    )

    # Elo and opponent management
    update_elo: bool = Field(True, description="Update Elo ratings after evaluation.")
    elo_registry_path: Optional[str] = Field(
        "elo_ratings.json", description="Path to Elo registry JSON file"
    )
    agent_id: Optional[str] = Field(
        None, description="Identifier for the evaluated model"
    )
    opponent_id: Optional[str] = Field(
        None, description="Identifier for the opponent model"
    )
    previous_model_pool_size: int = Field(
        5,
        description="Number of previous checkpoints to keep for Elo evaluation.",
    )

    # Performance optimization settings
    enable_in_memory_evaluation: bool = Field(
        True, description="Enable in-memory evaluation for better performance."
    )
    model_weight_cache_size: int = Field(
        5, description="Number of opponent model weights to cache in memory."
    )
    enable_parallel_execution: bool = Field(
        True, description="Enable parallel game execution."
    )
    process_restart_threshold: int = Field(
        100, description="Restart worker processes after N games."
    )
    temp_agent_device: str = Field(
        "cpu", description="Device for temporary agents during evaluation."
    )
    clear_cache_after_evaluation: bool = Field(
        True, description="Clear model weight cache after evaluation."
    )

    # Advanced evaluation features
    enable_background_tournaments: bool = Field(
        False,
        description="Enable background tournament system for continuous evaluation.",
    )
    enable_advanced_analytics: bool = Field(
        False, description="Enable advanced analytics and reporting."
    )
    enable_enhanced_opponents: bool = Field(
        False,
        description="Enable enhanced opponent management with adaptive selection.",
    )

    # Performance safeguards (required by Performance Engineer)
    max_evaluation_time_minutes: int = Field(
        30, description="Maximum time per evaluation run in minutes"
    )
    evaluation_timeout_per_game: int = Field(
        300, description="Timeout per individual game in seconds"
    )
    max_concurrent_evaluations: int = Field(
        1, description="Maximum concurrent evaluation processes"
    )
    enable_performance_monitoring: bool = Field(
        True, description="Enable performance monitoring and SLA validation"
    )
    memory_limit_mb: int = Field(1000, description="Maximum memory usage limit in MB")
    cpu_limit_percent: float = Field(
        80.0, description="Maximum CPU usage limit as percentage"
    )
    gpu_limit_percent: float = Field(
        80.0, description="Maximum GPU usage limit as percentage"
    )

    @field_validator("evaluation_interval_timesteps")
    # pylint: disable=no-self-argument
    def evaluation_interval_positive(cls, v):
        if v <= 0:
            raise ValueError("evaluation_interval_timesteps must be positive")
        return v

    @field_validator("num_games")
    # pylint: disable=no-self-argument
    def num_games_positive(cls, v):
        if v <= 0:
            raise ValueError("num_games must be positive")
        return v

    @field_validator("max_moves_per_game")
    # pylint: disable=no-self-argument
    def max_moves_positive(cls, v):
        if v <= 0:
            raise ValueError("max_moves_per_game must be positive")
        return v

    @field_validator("previous_model_pool_size")
    def pool_size_positive(cls, v):  # pylint: disable=no-self-argument
        if v <= 0:
            raise ValueError("previous_model_pool_size must be positive")
        return v

    @field_validator("strategy")
    def strategy_valid(cls, v):  # pylint: disable=no-self-argument
        if v not in VALID_EVALUATION_STRATEGIES:
            raise ValueError(f"strategy must be one of {VALID_EVALUATION_STRATEGIES}")
        return v

    @field_validator("max_concurrent_games")
    def max_concurrent_positive(cls, v):  # pylint: disable=no-self-argument
        if v <= 0:
            raise ValueError("max_concurrent_games must be positive")
        return v

    @field_validator("log_level")
    def log_level_valid(cls, v):  # pylint: disable=no-self-argument
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @field_validator("model_weight_cache_size")
    def cache_size_positive(cls, v):  # pylint: disable=no-self-argument
        if v <= 0:
            raise ValueError("model_weight_cache_size must be positive")
        return v

    @field_validator("process_restart_threshold")
    def restart_threshold_positive(cls, v):  # pylint: disable=no-self-argument
        if v <= 0:
            raise ValueError("process_restart_threshold must be positive")
        return v

    @field_validator("max_evaluation_time_minutes")
    def max_eval_time_positive(cls, v):  # pylint: disable=no-self-argument
        if v <= 0:
            raise ValueError("max_evaluation_time_minutes must be positive")
        return v

    @field_validator("evaluation_timeout_per_game")
    def eval_timeout_positive(cls, v):  # pylint: disable=no-self-argument
        if v <= 0:
            raise ValueError("evaluation_timeout_per_game must be positive")
        return v

    @field_validator("max_concurrent_evaluations")
    def max_concurrent_eval_positive(cls, v):  # pylint: disable=no-self-argument
        if v <= 0:
            raise ValueError("max_concurrent_evaluations must be positive")
        return v

    @field_validator("memory_limit_mb")
    def memory_limit_positive(cls, v):  # pylint: disable=no-self-argument
        if v <= 0:
            raise ValueError("memory_limit_mb must be positive")
        return v

    @field_validator("cpu_limit_percent")
    def cpu_limit_valid(cls, v):  # pylint: disable=no-self-argument
        if not 0 < v <= 100:
            raise ValueError("cpu_limit_percent must be between 0 and 100")
        return v

    @field_validator("gpu_limit_percent")
    def gpu_limit_valid(cls, v):  # pylint: disable=no-self-argument
        if not 0 < v <= 100:
            raise ValueError("gpu_limit_percent must be between 0 and 100")
        return v

    def get_strategy_param(self, key: str, default=None):
        """Get a strategy-specific parameter with optional default."""
        return self.strategy_params.get(key, default)

    def set_strategy_param(self, key: str, value) -> None:
        """Set a strategy-specific parameter."""
        self.strategy_params[key] = value

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return self.model_dump()

    def configure_for_single_opponent(
        self,
        opponent_name: str = "default_opponent",
        opponent_path: Optional[str] = None,
        play_as_both_colors: bool = True,
        color_balance_tolerance: float = 0.1,
        **kwargs,
    ) -> None:
        """Configure for single opponent strategy."""
        self.strategy = "single_opponent"
        self.strategy_params.update(
            {
                "opponent_name": opponent_name,
                "opponent_path": opponent_path,
                "play_as_both_colors": play_as_both_colors,
                "color_balance_tolerance": color_balance_tolerance,
                **kwargs,
            }
        )

    def configure_for_tournament(
        self,
        opponent_pool_config: Optional[list] = None,
        num_games_per_opponent: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Configure for tournament strategy."""
        self.strategy = "tournament"
        self.strategy_params.update(
            {
                "opponent_pool_config": opponent_pool_config or [],
                "num_games_per_opponent": num_games_per_opponent,
                **kwargs,
            }
        )

    def configure_for_ladder(
        self,
        opponent_pool_config: Optional[list] = None,
        elo_config: Optional[dict] = None,
        num_games_per_match: int = 2,
        num_opponents_per_evaluation: int = 3,
        rating_match_range: int = 200,
        **kwargs,
    ) -> None:
        """Configure for ladder strategy."""
        self.strategy = "ladder"
        self.strategy_params.update(
            {
                "opponent_pool_config": opponent_pool_config or [],
                "elo_config": elo_config or {},
                "num_games_per_match": num_games_per_match,
                "num_opponents_per_evaluation": num_opponents_per_evaluation,
                "rating_match_range": rating_match_range,
                **kwargs,
            }
        )

    def configure_for_benchmark(
        self,
        suite_config: Optional[list] = None,
        num_games_per_benchmark_case: int = 1,
        **kwargs,
    ) -> None:
        """Configure for benchmark strategy."""
        self.strategy = "benchmark"
        self.strategy_params.update(
            {
                "suite_config": suite_config or [],
                "num_games_per_benchmark_case": num_games_per_benchmark_case,
                **kwargs,
            }
        )


class LoggingConfig(BaseModel):
    log_file: str = Field(
        "logs/training_log.txt", description="Path for the main training log."
    )
    model_dir: str = Field(
        "models/", description="Directory to save model checkpoints."
    )
    run_name: Optional[str] = Field(
        None,
        description="Optional name for this run (overrides auto-generated name if set).",
    )


class WandBConfig(BaseModel):
    enabled: bool = Field(True, description="Enable Weights & Biases logging.")
    project: Optional[str] = Field("keisei-shogi-rl", description="W&B project name.")
    entity: Optional[str] = Field(None, description="W&B entity (username or team).")
    run_name_prefix: Optional[str] = Field(
        "keisei", description="Prefix for W&B run names."
    )
    watch_model: bool = Field(
        True, description="Use wandb.watch() to log model gradients and parameters."
    )
    watch_log_freq: int = Field(
        1000, description="Frequency for wandb.watch() logging."
    )
    watch_log_type: Literal["gradients", "parameters", "all"] = Field(
        "all", description="Type of data to log with wandb.watch()."
    )
    log_model_artifact: bool = Field(
        False, description="Enable logging model artifacts to W&B."
    )


class ParallelConfig(BaseModel):
    enabled: bool = Field(False, description="Enable parallel experience collection.")
    num_workers: int = Field(
        4, description="Number of parallel workers for experience collection."
    )
    batch_size: int = Field(
        32, description="Batch size for experience transmission from workers."
    )
    sync_interval: int = Field(
        100, description="Steps between model weight synchronization."
    )
    compression_enabled: bool = Field(
        True, description="Enable compression for model weight transmission."
    )
    timeout_seconds: float = Field(
        10.0, description="Timeout for worker communication operations."
    )
    max_queue_size: int = Field(1000, description="Maximum size of experience queues.")
    worker_seed_offset: int = Field(
        1000, description="Offset for worker random seeds to ensure diversity."
    )

    @field_validator("num_workers")
    # pylint: disable=no-self-argument
    def workers_positive(cls, v):
        if v <= 0:
            raise ValueError("num_workers must be positive")
        return v

    @field_validator("batch_size")
    # pylint: disable=no-self-argument
    def batch_size_positive(cls, v):
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v


class DemoConfig(BaseModel):
    enable_demo_mode: bool = Field(
        False,
        description="If True, enables demo mode with per-move delay and extra logging.",
    )
    demo_mode_delay: float = Field(
        0.5, description="Delay in seconds between moves in demo mode."
    )


class DisplayConfig(BaseModel):
    """Configuration for optional TUI display features."""

    enable_board_display: bool = Field(True, description="Show ASCII board panel")
    enable_trend_visualization: bool = Field(True, description="Show metric trends")
    enable_elo_ratings: bool = Field(True, description="Show Elo rating panel")
    enable_enhanced_layout: bool = Field(
        True, description="Use enhanced dashboard layout"
    )
    display_moves: bool = Field(
        False,
        description="Show full move descriptions and delay between turns",
    )
    turn_tick: float = Field(
        0.5,
        description="Delay in seconds between turns when display_moves is enabled",
    )
    board_unicode_pieces: bool = Field(True, description="Use Unicode pieces")
    board_cell_width: int = Field(5, description="Board cell width in characters")
    board_cell_height: int = Field(3, description="Board cell height in lines")
    board_highlight_last_move: bool = Field(True, description="Highlight last move")
    sparkline_width: int = Field(15, description="Sparkline width in characters")
    trend_history_length: int = Field(
        100, description="Number of history points to keep"
    )
    elo_initial_rating: float = Field(1500.0, description="Initial Elo rating")
    elo_k_factor: float = Field(32.0, description="Elo K-factor")
    dashboard_height_ratio: int = Field(2, description="Layout ratio for dashboard")
    progress_bar_height: int = Field(4, description="Progress bar height")
    show_text_moves: bool = Field(
        True,
        description="Display recent moves under the board when demo mode is active",
    )
    move_list_length: int = Field(10, description="Number of recent moves to display")
    moves_latest_top: bool = Field(
        True,
        description="Display newest move at top of recent moves panel",
    )
    moves_flash_ms: int = Field(
        500,
        description="Highlight newest move for N milliseconds (0 disables)",
    )
    show_moves_trend: bool = Field(True, description="Display moves per game trend")
    show_completion_rate: bool = Field(True, description="Display game completion rate")
    show_enhanced_win_rates: bool = Field(
        True, description="Display win/loss/draw breakdown"
    )
    show_turns_trend: bool = Field(True, description="Display average turns trend")
    metrics_window_size: int = Field(100, description="Rolling window size for metrics")
    trend_smoothing_factor: float = Field(
        0.1, description="Smoothing factor for trend arrows"
    )
    metrics_panel_height: int = Field(6, description="Height of metrics panel")
    enable_trendlines: bool = Field(True, description="Show trendlines in sparklines")
    log_layer_keyword_filters: List[str] = Field(
        ["stem", "policy_head", "value_head"],
        description=(
            "Keywords to filter layers in Model Evolution panel "
            "(layers containing any of these keywords will be displayed)"
        ),
    )


def _create_display_config() -> DisplayConfig:
    """Factory function for DisplayConfig to avoid lambda in default_factory."""
    return DisplayConfig()  # type: ignore[call-arg]


class WebUIConfig(BaseModel):
    """Configuration for WebUI streaming service for Twitch/demo streaming."""
    enabled: bool = Field(False, description="Enable WebUI streaming")
    port: int = Field(8765, description="WebUI server port")
    host: str = Field("0.0.0.0", description="WebUI server host (0.0.0.0 for all interfaces)")
    update_rate_hz: float = Field(
        2.0, description="Update frequency in Hz for real-time data"
    )
    max_connections: int = Field(
        10, description="Maximum concurrent WebSocket connections"
    )
    board_update_rate_hz: float = Field(
        5.0, description="Board state update frequency (higher for smoother moves)"
    )
    metrics_update_rate_hz: float = Field(
        1.0, description="Metrics update frequency"
    )


class AppConfig(BaseModel):
    env: EnvConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    wandb: WandBConfig
    parallel: ParallelConfig
    demo: Optional[DemoConfig] = None
    display: DisplayConfig = Field(default_factory=_create_display_config)
    webui: WebUIConfig = Field(default_factory=WebUIConfig)

    model_config = {"extra": "forbid"}  # Disallow unknown fields for strict validation
