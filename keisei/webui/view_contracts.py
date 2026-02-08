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

from typing import Any, Dict, List, Literal, NotRequired, Required, TypedDict, Union

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
    training: Required[Dict[str, Any]]  # TrainingViewState (eva.1.2)
    pending_updates: Required[Dict[str, ScalarValue]]

    # --- optional views (null until their Move delivers) -----------------
    league: NotRequired[Dict[str, Any] | None]
    lineage: NotRequired[Dict[str, Any] | None]
    skill_differential: NotRequired[Dict[str, Any] | None]
    model_profile: NotRequired[Dict[str, Any] | None]


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
