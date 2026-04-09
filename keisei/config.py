"""TOML config loading with dataclass validation."""

from __future__ import annotations

import math
import tomllib
import warnings
from dataclasses import dataclass, fields
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
    min_games_for_promotion: int = 64
    topk: int = 3
    streak_epochs: int = 50
    max_lineage_overlap: int = 2
    replace_policy: str = "weakest_or_stalest_after_cooldown"
    span_selection: bool = True

    def __post_init__(self) -> None:
        if self.slots < 1:
            raise ValueError(f"slots must be >= 1, got {self.slots}")
        if self.topk < 1:
            raise ValueError(f"topk must be >= 1, got {self.topk}")
        if self.review_interval_epochs < 1:
            raise ValueError(
                f"review_interval_epochs must be >= 1, got {self.review_interval_epochs}"
            )
        if self.min_games_for_promotion < 0:
            raise ValueError(
                f"min_games_for_promotion must be >= 0, got {self.min_games_for_promotion}"
            )
        if self.min_tenure_epochs < 0:
            raise ValueError(
                f"min_tenure_epochs must be >= 0, got {self.min_tenure_epochs}"
            )
        if self.replace_policy != "weakest_or_stalest_after_cooldown":
            raise ValueError(
                f"Only 'weakest_or_stalest_after_cooldown' replace_policy is supported, "
                f"got {self.replace_policy!r}"
            )


@dataclass(frozen=True)
class RecentFixedConfig:
    slots: int = 5
    min_games_for_review: int = 32
    min_unique_opponents: int = 6
    promotion_margin_elo: float = 25.0
    max_elo_spread: float = 200.0  # Plan §7.1 criterion 4: "acceptable uncertainty / volatility"
    soft_overflow: int = 1
    retire_if_below_dynamic_floor: bool = True

    def __post_init__(self) -> None:
        if self.slots < 1:
            raise ValueError(f"slots must be >= 1, got {self.slots}")
        if self.min_games_for_review < 0:
            raise ValueError(
                f"min_games_for_review must be >= 0, got {self.min_games_for_review}"
            )


@dataclass(frozen=True)
class DynamicConfig:
    slots: int = 10
    protection_matches: int = 24
    min_games_before_eviction: int = 40
    training_enabled: bool = True
    update_epochs_per_batch: int = 2
    lr_scale: float = 0.25
    grad_clip: float = 1.0  # must be > 0; PyTorch clips all grads to zero otherwise
    update_every_matches: int = 4
    max_updates_per_minute: int = 20
    checkpoint_flush_every: int = 8
    disable_on_error: bool = True
    max_buffer_depth: int = 8
    max_consecutive_errors: int = 3
    batch_reuse: int = 1
    global_error_threshold: int = 5
    global_error_window_seconds: float = 300.0
    gpu_memory_backpressure: float = 0.9

    def __post_init__(self) -> None:
        if self.protection_matches < 0:
            raise ValueError(
                f"protection_matches must be >= 0, got {self.protection_matches}"
            )
        if self.min_games_before_eviction < 0:
            raise ValueError(
                f"min_games_before_eviction must be >= 0, got {self.min_games_before_eviction}"
            )
        if self.update_epochs_per_batch < 1:
            raise ValueError(
                f"update_epochs_per_batch must be >= 1, got {self.update_epochs_per_batch}"
            )
        if self.grad_clip <= 0:
            raise ValueError(
                f"grad_clip must be > 0, got {self.grad_clip}"
            )
        if not (0 < self.lr_scale <= 1.0):
            raise ValueError(
                f"lr_scale must be in (0, 1.0], got {self.lr_scale}"
            )
        if self.update_every_matches < 1:
            raise ValueError(
                f"update_every_matches must be >= 1, got {self.update_every_matches}"
            )
        if self.max_updates_per_minute < 1:
            raise ValueError(
                f"max_updates_per_minute must be >= 1, got {self.max_updates_per_minute}"
            )
        if self.checkpoint_flush_every < 1:
            raise ValueError(
                f"checkpoint_flush_every must be >= 1, got {self.checkpoint_flush_every}"
            )
        if self.max_buffer_depth < 1:
            raise ValueError(
                f"max_buffer_depth must be >= 1, got {self.max_buffer_depth}"
            )
        if self.max_consecutive_errors < 1:
            raise ValueError(
                f"max_consecutive_errors must be >= 1, got {self.max_consecutive_errors}"
            )
        if self.batch_reuse < 1:
            raise ValueError(
                f"batch_reuse must be >= 1, got {self.batch_reuse}"
            )
        if self.batch_reuse > 1:
            warnings.warn(
                f"batch_reuse={self.batch_reuse} is configured but not yet "
                "implemented — the value will be ignored. Only "
                "update_epochs_per_batch controls training iterations.",
                stacklevel=2,
            )
        if self.global_error_threshold < 1:
            raise ValueError(
                f"global_error_threshold must be >= 1, got {self.global_error_threshold}"
            )
        if self.global_error_window_seconds <= 0:
            raise ValueError(
                f"global_error_window_seconds must be > 0, got {self.global_error_window_seconds}"
            )
        if not (0 < self.gpu_memory_backpressure <= 1.0):
            raise ValueError(
                f"gpu_memory_backpressure must be in (0, 1.0], got {self.gpu_memory_backpressure}"
            )


@dataclass(frozen=True)
class MatchSchedulerConfig:
    learner_dynamic_ratio: float = 0.50
    learner_frontier_ratio: float = 0.30
    learner_recent_ratio: float = 0.20
    tournament_games_per_pair: int = 3  # best-of-3 round-robin
    tournament_mode: str = "full"  # "full", "weighted", or "random"
    weighted_round_size: int = 0  # 0 = auto (N entries → N pairings per round)
    pairing_policy: str = "role_weighted_sparse_h2h"
    # Match-class weights for weighted tournament mode (§8.3).
    dynamic_dynamic_weight: float = 0.40
    dynamic_recent_weight: float = 0.25
    dynamic_frontier_weight: float = 0.20
    recent_frontier_weight: float = 0.10
    recent_recent_weight: float = 0.05
    # Challenge threshold: when the learner's rolling win rate against a tier
    # exceeds this value, halve that tier's sampling weight so the learner
    # trains more against tiers it hasn't yet mastered.
    challenge_threshold: float = 0.70
    challenge_window: int = 100  # rolling window of recent results per tier

    def __post_init__(self) -> None:
        for name in ("learner_dynamic_ratio", "learner_frontier_ratio", "learner_recent_ratio"):
            val = getattr(self, name)
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")
        learner_sum = (
            self.learner_dynamic_ratio
            + self.learner_frontier_ratio
            + self.learner_recent_ratio
        )
        if abs(learner_sum - 1.0) > 1e-6:
            raise ValueError(
                f"learner mix ratio sum must be 1.0, got {learner_sum}"
            )
        match_weight_sum = (
            self.dynamic_dynamic_weight
            + self.dynamic_recent_weight
            + self.dynamic_frontier_weight
            + self.recent_frontier_weight
            + self.recent_recent_weight
        )
        if abs(match_weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"match-class weight sum must be 1.0, got {match_weight_sum}"
            )
        if self.tournament_games_per_pair < 1:
            raise ValueError(
                f"tournament_games_per_pair must be >= 1, got {self.tournament_games_per_pair}"
            )
        valid_modes = ("full", "weighted", "random")
        if self.tournament_mode not in valid_modes:
            raise ValueError(
                f"tournament_mode must be one of {valid_modes}, got {self.tournament_mode!r}"
            )
        if self.weighted_round_size < 0:
            raise ValueError(
                f"weighted_round_size must be >= 0, got {self.weighted_round_size}"
            )
        if self.pairing_policy != "role_weighted_sparse_h2h":
            raise ValueError(
                f"Only 'role_weighted_sparse_h2h' pairing_policy is supported, "
                f"got {self.pairing_policy!r}"
            )


@dataclass(frozen=True)
class HistoricalLibraryConfig:
    enabled: bool = True
    slots: int = 5
    refresh_interval_epochs: int = 100
    min_epoch_for_selection: int = 10
    selection: str = "log_spaced"
    active_league_participation: bool = False

    def __post_init__(self) -> None:
        if self.slots < 1:
            raise ValueError(f"slots must be >= 1, got {self.slots}")
        if self.refresh_interval_epochs < 1:
            raise ValueError(
                f"refresh_interval_epochs must be >= 1, got {self.refresh_interval_epochs}"
            )
        if self.selection != "log_spaced":
            raise ValueError(
                f"Only 'log_spaced' selection is supported, got {self.selection!r}"
            )
        if self.active_league_participation:
            raise ValueError(
                "active_league_participation must be false — historical library "
                "entries do not participate in active-league matchmaking"
            )


@dataclass(frozen=True)
class GauntletConfig:
    enabled: bool = True
    interval_epochs: int = 100
    games_per_matchup: int = 16

    def __post_init__(self) -> None:
        if self.interval_epochs < 1:
            raise ValueError(
                f"interval_epochs must be >= 1, got {self.interval_epochs}"
            )
        if self.games_per_matchup < 1:
            raise ValueError(
                f"games_per_matchup must be >= 1, got {self.games_per_matchup}"
            )


@dataclass(frozen=True)
class RoleEloConfig:
    frontier_k: float = 16.0
    dynamic_k: float = 24.0
    recent_k: float = 32.0
    historical_k: float = 12.0
    track_role_specific: bool = True

    def __post_init__(self) -> None:
        for name in ("frontier_k", "dynamic_k", "recent_k", "historical_k"):
            val = getattr(self, name)
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")
        if not self.track_role_specific:
            raise ValueError(
                "track_role_specific must be true — role-specific Elo tracking "
                "is required for the tiered league to function correctly"
            )


@dataclass(frozen=True)
class PriorityScorerConfig:
    under_sample_weight: float = 1.0
    uncertainty_weight: float = 0.5
    recent_fixed_bonus: float = 0.3
    diversity_weight: float = 0.3
    match_class_weight: float = 1.0
    frontier_exposure_weight: float = 0.4
    frontier_exposure_threshold: int = 10
    repeat_penalty: float = -0.5
    lineage_penalty: float = -0.3
    repeat_window_rounds: int = 5

    def __post_init__(self) -> None:
        for field_name in (
            "under_sample_weight",
            "uncertainty_weight",
            "recent_fixed_bonus",
            "diversity_weight",
            "match_class_weight",
            "frontier_exposure_weight",
            "repeat_penalty",
            "lineage_penalty",
        ):
            val = getattr(self, field_name)
            if not math.isfinite(val):
                raise ValueError(f"{field_name} must be finite, got {val}")
        for penalty_name in ("repeat_penalty", "lineage_penalty"):
            val = getattr(self, penalty_name)
            if val > 0:
                raise ValueError(
                    f"{penalty_name} must be <= 0 (penalty, not bonus), got {val}"
                )
        if self.repeat_window_rounds < 1:
            raise ValueError(
                f"repeat_window_rounds must be >= 1, got {self.repeat_window_rounds}"
            )


@dataclass(frozen=True)
class ConcurrencyConfig:
    parallel_matches: int = 4
    envs_per_match: int = 8
    total_envs: int = 32
    max_resident_models: int = 10

    def __post_init__(self) -> None:
        needed_envs = self.parallel_matches * self.envs_per_match
        if needed_envs > self.total_envs:
            raise ValueError(
                f"parallel_matches * envs_per_match ({needed_envs}) "
                f"exceeds total_envs ({self.total_envs})"
            )
        if self.max_resident_models < 2:
            raise ValueError(
                f"max_resident_models ({self.max_resident_models}) must be >= 2 "
                f"(at least one model pair)"
            )
    @property
    def effective_parallel(self) -> int:
        """Max concurrent slots — equals parallel_matches.

        Model sharing via the LRU cache means slots don't need 2 unique
        models each.  Cache sizing is validated at LeagueConfig level.
        """
        return self.parallel_matches


@dataclass(frozen=True)
class StorageConfig:
    """Controls checkpoint storage semantics for league entries."""

    clone_on_promotion: bool = True
    persist_optimizer_for_dynamic: bool = True

    def __post_init__(self) -> None:
        if not self.clone_on_promotion:
            raise ValueError(
                "clone_on_promotion must be true — promotion by cloning "
                "is required to preserve lineage semantics"
            )
        if not self.persist_optimizer_for_dynamic:
            raise ValueError(
                "persist_optimizer_for_dynamic must be true — Dynamic entries "
                "require persistent optimizer state for training continuity"
            )


@dataclass(frozen=True)
class LeagueConfig:
    enabled: bool = True
    mode: str = "mixed"
    max_active_entries: int | None = None
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
    tournament_games_per_match: int = 3
    tournament_k_factor: float = 16.0
    tournament_pause_seconds: float = 1.0
    # Mutable-default-argument pitfall does NOT apply here: every sub-config
    # dataclass is frozen=True, so shared default instances can't be mutated.
    frontier: FrontierStaticConfig = FrontierStaticConfig()
    recent: RecentFixedConfig = RecentFixedConfig()
    dynamic: DynamicConfig = DynamicConfig()
    scheduler: MatchSchedulerConfig = MatchSchedulerConfig()
    history: HistoricalLibraryConfig = HistoricalLibraryConfig()
    gauntlet: GauntletConfig = GauntletConfig()
    elo: RoleEloConfig = RoleEloConfig()
    priority: PriorityScorerConfig = PriorityScorerConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    storage: StorageConfig = StorageConfig()

    def __post_init__(self) -> None:
        # These validate LeagueConfig's OWN scalar fields — none overlap with
        # sub-config validation.  In particular, tournament_games_per_match
        # (total games in a tournament match) is distinct from
        # MatchSchedulerConfig.tournament_games_per_pair (round-robin pair count).
        if self.mode != "mixed":
            raise ValueError(
                f"Only 'mixed' league mode is supported, got {self.mode!r}"
            )
        if self.epochs_per_seat < 1:
            raise ValueError(
                f"league.epochs_per_seat must be >= 1, got {self.epochs_per_seat}"
            )
        if self.snapshot_interval < 1:
            raise ValueError(
                f"league.snapshot_interval must be >= 1, got {self.snapshot_interval}"
            )
        if self.elo_floor > self.initial_elo:
            raise ValueError(
                f"elo_floor ({self.elo_floor}) must be <= initial_elo ({self.initial_elo})"
            )
        if self.tournament_games_per_match < 1:
            raise ValueError(
                f"tournament_games_per_match must be >= 1, got {self.tournament_games_per_match}"
            )
        if self.elo_k_factor <= 0:
            raise ValueError(
                f"elo_k_factor must be > 0, got {self.elo_k_factor}"
            )
        if self.tournament_k_factor <= 0:
            raise ValueError(
                f"tournament_k_factor must be > 0, got {self.tournament_k_factor}"
            )
        if self.max_active_entries is not None and self.max_active_entries < 1:
            raise ValueError(
                f"max_active_entries must be >= 1 or None, got {self.max_active_entries}"
            )
        # Cross-config validation: warn if LRU cache can't hold the full pool
        if (
            self.max_active_entries is not None
            and self.concurrency.max_resident_models < self.max_active_entries
        ):
            warnings.warn(
                f"max_resident_models ({self.concurrency.max_resident_models}) < "
                f"max_active_entries ({self.max_active_entries}): LRU model cache "
                f"cannot hold the full opponent pool, which may cause GPU memory "
                f"thrashing during concurrent matches",
                stacklevel=2,
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
    _valid_training_keys = {f.name for f in fields(TrainingConfig)}
    _unknown_training = set(t.keys()) - _valid_training_keys
    if _unknown_training:
        raise ValueError(
            f"Unknown [training] config keys: {sorted(_unknown_training)}. "
            f"Valid keys: {sorted(_valid_training_keys)}"
        )
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
    _valid_display_keys = {f.name for f in fields(DisplayConfig)}
    _unknown_display = set(d.keys()) - _valid_display_keys
    if _unknown_display:
        raise ValueError(
            f"Unknown [display] config keys: {sorted(_unknown_display)}. "
            f"Valid keys: {sorted(_valid_display_keys)}"
        )
    moves_per_minute = d.get("moves_per_minute", 30)
    if moves_per_minute < 0:
        raise ValueError(f"moves_per_minute must be >= 0, got {moves_per_minute}")
    db_path = str((config_dir / d.get("db_path", "keisei.db")).resolve())

    display = DisplayConfig(moves_per_minute=moves_per_minute, db_path=db_path)

    m = raw.get("model", {})
    _valid_model_keys = {f.name for f in fields(ModelConfig)}
    _unknown_model = set(m.keys()) - _valid_model_keys
    if _unknown_model:
        raise ValueError(
            f"Unknown [model] config keys: {sorted(_unknown_model)}. "
            f"Valid keys: {sorted(_valid_model_keys)}"
        )
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

        # --- Reject old TOML section names ---
        _removed_sections = {
            "frontier_static": "frontier",
            "recent_fixed": "recent",
            "sampling": "scheduler",
            "role_elo": "elo",
            "matchmaking": "concurrency] and [league.priority",
        }
        for old_name, new_name in _removed_sections.items():
            if old_name in lg:
                raise ValueError(
                    f"[league.{old_name}] is not a valid config section. "
                    f"Use [league.{new_name}] instead."
                )

        # --- Extract sub-config sections (1:1 mapping to LeagueConfig attributes) ---
        frontier_raw = lg.pop("frontier", {})
        recent_raw = lg.pop("recent", {})
        dynamic_raw = lg.pop("dynamic", {})
        scheduler_raw = lg.pop("scheduler", {})
        history_raw = lg.pop("history", {})
        gauntlet_raw = lg.pop("gauntlet", {})
        elo_raw = lg.pop("elo", {})
        priority_raw = lg.pop("priority", {})
        concurrency_raw = lg.pop("concurrency", {})
        storage_raw = lg.pop("storage", {})

        # [league.active] is convenience sugar — slot counts merge into sub-configs
        active_raw = lg.pop("active", {})
        if active_raw:
            if "frontier_static_slots" in active_raw:
                frontier_raw.setdefault("slots", active_raw["frontier_static_slots"])
            if "recent_fixed_slots" in active_raw:
                recent_raw.setdefault("slots", active_raw["recent_fixed_slots"])
            if "dynamic_slots" in active_raw:
                dynamic_raw.setdefault("slots", active_raw["dynamic_slots"])

        # --- Legacy keys from pre-tiered-pool era ---
        _legacy_league_keys = {"max_pool_size", "historical_ratio", "current_best_ratio"}
        found_legacy = _legacy_league_keys & set(lg.keys())
        if found_legacy:
            warnings.warn(
                f"Ignoring deprecated [league] keys: {sorted(found_legacy)}. "
                f"These were removed during the tiered-pool migration.",
                DeprecationWarning,
                stacklevel=2,
            )
        for key in _legacy_league_keys:
            lg.pop(key, None)

        # --- Validate remaining keys against LeagueConfig scalar fields ---
        _sub_config_names = {
            "frontier", "recent", "dynamic", "scheduler", "history",
            "gauntlet", "elo", "priority", "concurrency", "storage", "active",
        }
        valid_league_keys = {
            f.name for f in fields(LeagueConfig)
        } - _sub_config_names
        unknown_league = set(lg.keys()) - valid_league_keys
        if unknown_league:
            raise ValueError(
                f"Unknown [league] config keys: {sorted(unknown_league)}. "
                f"Valid keys: {sorted(valid_league_keys)}"
            )
        league_config = LeagueConfig(
            **lg,
            frontier=FrontierStaticConfig(**frontier_raw),
            recent=RecentFixedConfig(**recent_raw),
            dynamic=DynamicConfig(**dynamic_raw),
            scheduler=MatchSchedulerConfig(**scheduler_raw),
            history=HistoricalLibraryConfig(**history_raw),
            gauntlet=GauntletConfig(**gauntlet_raw),
            elo=RoleEloConfig(**elo_raw),
            priority=PriorityScorerConfig(**priority_raw),
            concurrency=ConcurrencyConfig(**concurrency_raw),
            storage=StorageConfig(**storage_raw),
        )
        # If league is explicitly disabled, treat as absent — all existing
        # `if config.league is not None` checks continue to work.
        if not league_config.enabled:
            league_config = None

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
