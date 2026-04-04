"""TOML config loading with dataclass validation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from keisei.training.algorithm_registry import VALID_ALGORITHMS  # noqa: E402

# Import valid sets from the registries — single source of truth.
# This eliminates the "two truths" problem where config.py and the registries
# could independently diverge. Adding a new architecture or algorithm to the
# registry automatically makes it available in config validation.
from keisei.training.model_registry import VALID_ARCHITECTURES  # noqa: E402


@dataclass(frozen=True)
class TrainingConfig:
    num_games: int
    max_ply: int
    algorithm: str
    checkpoint_interval: int
    checkpoint_dir: str
    algorithm_params: dict[str, Any]
    use_amp: bool = False


@dataclass(frozen=True)
class DisplayConfig:
    moves_per_minute: int
    db_path: str


@dataclass(frozen=True)
class ModelConfig:
    display_name: str
    architecture: str
    params: dict[str, object]


@dataclass(frozen=True)
class FrontierStaticConfig:
    slots: int = 5
    review_interval_epochs: int = 250
    min_tenure_epochs: int = 100
    promotion_margin_elo: float = 50.0


@dataclass(frozen=True)
class RecentFixedConfig:
    slots: int = 5
    min_games_for_review: int = 32
    min_unique_opponents: int = 6
    promotion_margin_elo: float = 25.0
    soft_overflow: int = 1


@dataclass(frozen=True)
class DynamicConfig:
    slots: int = 10
    protection_matches: int = 24
    min_games_before_eviction: int = 40
    training_enabled: bool = False


@dataclass(frozen=True)
class MatchSchedulerConfig:
    learner_dynamic_ratio: float = 0.50
    learner_frontier_ratio: float = 0.30
    learner_recent_ratio: float = 0.20
    tournament_games_per_pair: int = 3  # best-of-3 round-robin


@dataclass(frozen=True)
class LeagueConfig:
    snapshot_interval: int = 10
    epochs_per_seat: int = 50
    initial_elo: float = 1000.0
    elo_k_factor: float = 32.0
    elo_floor: float = 500.0
    color_randomization: bool = True
    per_env_opponents: bool = True
    opponent_device: str | None = None
    tournament_enabled: bool = False
    tournament_device: str | None = None
    tournament_num_envs: int = 64
    tournament_games_per_match: int = 64
    tournament_k_factor: float = 16.0
    tournament_pause_seconds: float = 5.0
    frontier: FrontierStaticConfig = FrontierStaticConfig()
    recent: RecentFixedConfig = RecentFixedConfig()
    dynamic: DynamicConfig = DynamicConfig()
    scheduler: MatchSchedulerConfig = MatchSchedulerConfig()

    def __post_init__(self) -> None:
        if self.epochs_per_seat < 1:
            raise ValueError(
                f"league.epochs_per_seat must be >= 1, got {self.epochs_per_seat}"
            )
        if self.snapshot_interval < 1:
            raise ValueError(
                f"league.snapshot_interval must be >= 1, got {self.snapshot_interval}"
            )
        s = self.scheduler
        learner_sum = s.learner_dynamic_ratio + s.learner_frontier_ratio + s.learner_recent_ratio
        if abs(learner_sum - 1.0) > 1e-6:
            raise ValueError(
                f"learner mix ratio sum must be 1.0, got {learner_sum}"
            )
        if self.scheduler.tournament_games_per_pair < 1:
            raise ValueError(
                f"tournament_games_per_pair must be >= 1, got {self.scheduler.tournament_games_per_pair}"
            )


@dataclass(frozen=True)
class DemonstratorConfig:
    num_games: int = 3
    auto_matchup: bool = True
    moves_per_minute: int = 60
    device: str = "cuda"


@dataclass(frozen=True)
class DistributedConfig:
    """Configuration for PyTorch DDP multi-GPU training.

    DDP activation is determined by torchrun environment variables (RANK,
    LOCAL_RANK, WORLD_SIZE), not by this config. This config controls DDP
    behavior only when DDP is active.
    """

    sync_batchnorm: bool = True
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True


@dataclass(frozen=True)
class AppConfig:
    training: TrainingConfig
    display: DisplayConfig
    model: ModelConfig
    league: LeagueConfig | None = None
    demonstrator: DemonstratorConfig | None = None
    distributed: DistributedConfig = DistributedConfig()


def load_config(path: Path) -> AppConfig:
    """Load and validate a TOML config file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    config_dir = path.parent.resolve()

    t = raw.get("training", {})
    num_games = t.get("num_games", 8)
    if not (1 <= num_games <= 512):
        raise ValueError(f"num_games must be 1-512, got {num_games}")

    max_ply = t.get("max_ply", 500)
    if max_ply <= 0:
        raise ValueError(f"max_ply must be positive, got {max_ply}")

    algorithm = t.get("algorithm", "katago_ppo")
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Valid: {sorted(VALID_ALGORITHMS)}"
        )

    checkpoint_interval = t.get("checkpoint_interval", 50)
    if checkpoint_interval <= 0:
        raise ValueError(
            f"checkpoint_interval must be positive, got {checkpoint_interval}"
        )

    checkpoint_dir = str(
        (config_dir / t.get("checkpoint_dir", "checkpoints/")).resolve()
    )
    algorithm_params = t.get("algorithm_params", {})
    use_amp_raw = t.get("use_amp", False)
    if not isinstance(use_amp_raw, bool):
        raise ValueError(
            f"training.use_amp must be a boolean, got {type(use_amp_raw).__name__}"
        )
    use_amp = use_amp_raw

    training = TrainingConfig(
        num_games=num_games,
        max_ply=max_ply,
        algorithm=algorithm,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        algorithm_params=algorithm_params,
        use_amp=use_amp,
    )

    d = raw.get("display", {})
    moves_per_minute = d.get("moves_per_minute", 30)
    if moves_per_minute < 0:
        raise ValueError(f"moves_per_minute must be >= 0, got {moves_per_minute}")
    db_path = str((config_dir / d.get("db_path", "keisei.db")).resolve())

    display = DisplayConfig(moves_per_minute=moves_per_minute, db_path=db_path)

    m = raw.get("model", {})
    display_name = m.get("display_name", "Player")
    architecture = m.get("architecture", "resnet")
    if architecture not in VALID_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Valid: {sorted(VALID_ARCHITECTURES)}"
        )
    model_params = m.get("params", {})

    model = ModelConfig(
        display_name=display_name, architecture=architecture, params=model_params
    )

    league_config = None
    if "league" in raw:
        lg = dict(raw["league"])
        frontier_raw = lg.pop("frontier", {})
        recent_raw = lg.pop("recent", {})
        dynamic_raw = lg.pop("dynamic", {})
        scheduler_raw = lg.pop("scheduler", {})
        # Strip legacy keys removed during tiered-pool migration
        _legacy_league_keys = {"max_pool_size", "historical_ratio", "current_best_ratio"}
        for key in _legacy_league_keys:
            lg.pop(key, None)
        league_config = LeagueConfig(
            **lg,
            frontier=FrontierStaticConfig(**frontier_raw),
            recent=RecentFixedConfig(**recent_raw),
            dynamic=DynamicConfig(**dynamic_raw),
            scheduler=MatchSchedulerConfig(**scheduler_raw),
        )

    demo_config = None
    if "demonstrator" in raw:
        demo_config = DemonstratorConfig(**raw["demonstrator"])

    dist_raw = raw.get("distributed", {})
    valid_dist_fields = {"sync_batchnorm", "find_unused_parameters", "gradient_as_bucket_view"}
    unknown = set(dist_raw.keys()) - valid_dist_fields
    if unknown:
        raise ValueError(
            f"Unknown [distributed] config keys: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid_dist_fields)}"
        )
    for key in sorted(valid_dist_fields):
        if key in dist_raw and not isinstance(dist_raw[key], bool):
            raise ValueError(
                f"distributed.{key} must be a boolean, got "
                f"{type(dist_raw[key]).__name__}"
            )
    dist_config = DistributedConfig(**dist_raw)

    return AppConfig(
        training=training, display=display, model=model,
        league=league_config, demonstrator=demo_config,
        distributed=dist_config,
    )
