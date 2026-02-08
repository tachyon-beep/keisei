"""
v1 Broadcast View Contracts — typed contract definitions for the broadcast
state envelope and view payloads.

These types define the schema for the JSON state file shared between the
training producer (state_snapshot.py / StreamlitManager) and the rendering
consumer (streamlit_app.py).

Schema versioning policy (Decision Freeze #1):
    - schema_version: semver-like string, currently "v1.0.0".
    - Patch-level bumps: optional field additions only.
    - Minor/major bumps: required-field or semantic changes; each requires a
      migration note in the spike doc.

Required envelope fields (must always be present):
    schema_version, timestamp, speed, mode, active_views, health, training,
    pending_updates.

Optional envelope fields (null when the producing Move has not delivered):
    league, lineage, skill_differential, model_profile.

Health semantics (Decision Freeze #2):
    Each view category has a health status in the envelope health map.
    Statuses: ok, stale, missing, error.
    Producer-side stale threshold: 30 seconds.
    Current snapshot sections (board_state, metrics, step_info, buffer_info,
    model_info) are sub-keys of the *training* view payload — they are NOT
    independent health-tracked views.

Mode semantics (Decision Freeze #6):
    mode reflects the active evaluation strategy configuration.  Values are
    aligned with EvaluationStrategy in config_schema.py, plus "training_only"
    for evaluation-disabled runs.  Unknown mode values must be treated as
    opaque strings by the renderer.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Literal,
    NotRequired,
    Optional,
    Required,
    TypedDict,
    Union,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: str = "v1.0.0"
"""Current broadcast schema version (semver-like)."""

STALE_THRESHOLD_SECONDS: float = 30.0
"""Default producer-side stale threshold matching current UI behaviour."""

VIEW_KEYS: tuple[str, ...] = (
    "training",
    "league",
    "lineage",
    "skill_differential",
    "model_profile",
)
"""Canonical view category names that appear in the health map."""

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

HealthStatus = Literal["ok", "stale", "missing", "error"]
"""Status of a single view category in the health map."""

BroadcastMode = Literal[
    "training_only",
    "single_opponent",
    "tournament",
    "ladder",
    "benchmark",
    "custom",
]
"""Frozen v1 mode values.  Renderers must treat unknown strings gracefully."""

ScalarValue = Union[int, float, str, bool, None]
"""Allowed value types inside ``pending_updates`` (Decision Freeze #3)."""

HealthMap = Dict[str, HealthStatus]
"""Mapping of each VIEW_KEYS entry to its current HealthStatus."""

# ---------------------------------------------------------------------------
# Training view sub-types (derived from state_snapshot.py extractors)
# ---------------------------------------------------------------------------


class PieceInfo(TypedDict):
    """A single piece on the board."""

    type: str  # e.g. "king", "pawn", "promoted_rook"
    color: str  # "black" or "white"
    promoted: bool


class BoardState(TypedDict):
    """9x9 board with hands and game status.

    Produced by ``extract_board_state()``.  Always present when a game is
    active; ``None`` between episodes.
    """

    board: List[List[Optional[PieceInfo]]]  # 9 rows x 9 cols
    current_player: str
    move_count: int
    game_over: bool
    winner: Optional[str]
    black_hand: Dict[str, int]
    white_hand: Dict[str, int]


class LearningCurves(TypedDict):
    """Trailing history of PPO training metrics (last 50 values)."""

    policy_losses: List[float]
    value_losses: List[float]
    entropies: List[float]
    kl_divergences: List[float]
    clip_fractions: List[float]
    learning_rates: List[float]
    episode_lengths: List[float]
    episode_rewards: List[float]


class MetricsState(TypedDict):
    """Aggregate training metrics.

    Produced by ``extract_metrics()``.  Always present.
    """

    global_timestep: int
    total_episodes: int
    black_wins: int
    white_wins: int
    draws: int
    processing: bool
    learning_curves: LearningCurves
    win_rates_history: List[Dict[str, float]]
    hot_squares: Any  # opaque structure from MetricsManager.get_hot_squares


class StepInfo(TypedDict):
    """Per-episode step and move statistics.

    Produced by ``extract_step_info()``.  ``None`` before the first episode.
    """

    move_log: List[str]
    sente_capture_count: int
    gote_capture_count: int
    sente_drop_count: int
    gote_drop_count: int
    sente_promo_count: int
    gote_promo_count: int


class BufferInfo(TypedDict):
    """Experience buffer fill level.

    ``None`` before the buffer is initialised.
    """

    size: int
    capacity: int


class ModelInfo(TypedDict):
    """Model-level training signals."""

    gradient_norm: float


# ---------------------------------------------------------------------------
# Per-view state schemas
# ---------------------------------------------------------------------------


class TrainingViewState(TypedDict):
    """Training view payload — the only populated view in v1.

    Sub-keys map 1:1 to the current ``build_snapshot()`` output per the
    frozen key mapping table in the spike doc.

    Invariants:
        - ``metrics`` and ``model_info`` are always present (never None).
        - ``board_state`` is None between episodes.
        - ``step_info`` is None before the first episode starts.
        - ``buffer_info`` is None before the experience buffer is created.
    """

    board_state: Optional[BoardState]
    metrics: MetricsState
    step_info: Optional[StepInfo]
    buffer_info: Optional[BufferInfo]
    model_info: ModelInfo


class LeagueViewState(TypedDict):
    """League / Elo snapshot view — placeholder, produced by Move 2.

    Shape derived from the ``evaluation_elo_snapshot`` dict in
    ``callbacks.py``.  Not populated until ``keisei-tw5`` delivers a
    producer.
    """

    current_id: str
    current_rating: float
    opponent_id: str
    opponent_rating: float
    last_outcome: str  # "win", "loss", or "draw"
    top_ratings: List[List[Any]]  # [[id, rating], ...]


class LineageViewState(TypedDict):
    """Lineage / provenance view — placeholder, produced by Move 2.

    Backed by append-only JSONL events (Decision Freeze #4).  Not populated
    until ``keisei-tw5`` delivers event persistence and a read model.
    """

    event_count: int
    latest_checkpoint_id: Optional[str]
    parent_id: Optional[str]


class SkillDifferentialViewState(TypedDict):
    """Skill differential view — placeholder for future implementation.

    Intended to show comparative skill metrics across evaluation runs.
    """

    placeholder: bool  # always True; real fields TBD


class ModelProfileViewState(TypedDict):
    """Model profile view — placeholder for future implementation.

    Intended to expose architecture metadata and parameter counts.
    """

    placeholder: bool  # always True; real fields TBD


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


class BroadcastStateEnvelope(TypedDict, total=False):
    """Top-level broadcast state envelope (v1).

    *Required* fields are always present in every snapshot.  *Optional* view
    fields are ``None`` until the producing Move delivers a producer for that
    view category.

    The ``training`` view is the only populated view in v1.  Its internal
    structure is defined by ``TrainingViewState`` (see eva.1.2).
    """

    # --- required --------------------------------------------------------
    schema_version: Required[str]
    timestamp: Required[float]
    speed: Required[float]
    mode: Required[str]  # BroadcastMode at runtime, str for forward compat
    active_views: Required[List[str]]
    health: Required[HealthMap]
    training: Required[TrainingViewState]
    pending_updates: Required[Dict[str, ScalarValue]]

    # --- optional views (null until their Move delivers) -----------------
    league: NotRequired[LeagueViewState | None]
    lineage: NotRequired[LineageViewState | None]
    skill_differential: NotRequired[SkillDifferentialViewState | None]
    model_profile: NotRequired[ModelProfileViewState | None]


def make_health_map(**overrides: HealthStatus) -> HealthMap:
    """Build a full health map with ``missing`` as the default for each view.

    >>> make_health_map(training="ok")
    {'training': 'ok', 'league': 'missing', ...}
    """
    base: HealthMap = {k: "missing" for k in VIEW_KEYS}
    for key, status in overrides.items():
        if key not in base:
            raise ValueError(
                f"Unknown view key {key!r}; valid keys are {VIEW_KEYS}"
            )
        base[key] = status
    return base


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_HEALTH_STATUSES = {"ok", "stale", "missing", "error"}

_REQUIRED_ENVELOPE_KEYS = {
    "schema_version",
    "timestamp",
    "speed",
    "mode",
    "active_views",
    "health",
    "training",
    "pending_updates",
}


def validate_envelope(data: Dict[str, Any]) -> List[str]:
    """Check *data* against the ``BroadcastStateEnvelope`` contract.

    Returns a list of human-readable error strings.  An empty list means
    the envelope is valid.  This is a structural check only — it does not
    deep-validate individual view payloads.
    """
    errors: List[str] = []

    # Required keys
    for key in _REQUIRED_ENVELOPE_KEYS:
        if key not in data:
            errors.append(f"missing required key: {key!r}")

    if errors:
        return errors  # can't check further without required keys

    # schema_version
    sv = data["schema_version"]
    if not isinstance(sv, str) or not sv:
        errors.append(f"schema_version must be a non-empty string, got {sv!r}")

    # timestamp / speed
    for num_key in ("timestamp", "speed"):
        val = data[num_key]
        if not isinstance(val, (int, float)):
            errors.append(f"{num_key} must be numeric, got {type(val).__name__}")

    # mode
    if not isinstance(data["mode"], str):
        errors.append(f"mode must be a string, got {type(data['mode']).__name__}")

    # active_views
    av = data["active_views"]
    if not isinstance(av, list) or not all(isinstance(v, str) for v in av):
        errors.append("active_views must be a list of strings")

    # health map completeness
    health = data["health"]
    if not isinstance(health, dict):
        errors.append(f"health must be a dict, got {type(health).__name__}")
    else:
        for vk in VIEW_KEYS:
            if vk not in health:
                errors.append(f"health map missing view key: {vk!r}")
            elif health[vk] not in _VALID_HEALTH_STATUSES:
                errors.append(
                    f"health[{vk!r}] has invalid status {health[vk]!r}; "
                    f"expected one of {_VALID_HEALTH_STATUSES}"
                )

    # training must be a dict (deep validation is per-view)
    if not isinstance(data["training"], dict):
        errors.append(
            f"training must be a dict, got {type(data['training']).__name__}"
        )

    # pending_updates scalar check
    pu = data["pending_updates"]
    if not isinstance(pu, dict):
        errors.append(
            f"pending_updates must be a dict, got {type(pu).__name__}"
        )
    else:
        for pk, pv in pu.items():
            if not isinstance(pv, (int, float, str, bool, type(None))):
                errors.append(
                    f"pending_updates[{pk!r}] has non-scalar value "
                    f"of type {type(pv).__name__}"
                )

    return errors


def sanitize_pending_updates(
    raw: Dict[str, Any] | None,
) -> Dict[str, ScalarValue]:
    """Filter *raw* to scalar-only values per Decision Freeze #3.

    Non-scalar values are silently dropped — never fatal to snapshot writes.
    Returns an empty dict when *raw* is ``None`` or empty.
    """
    if not raw:
        return {}
    return {
        k: v
        for k, v in raw.items()
        if isinstance(v, (int, float, str, bool, type(None)))
    }
