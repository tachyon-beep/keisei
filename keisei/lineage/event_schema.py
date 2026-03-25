"""
Lineage event schema — typed definitions for the append-only event log.

Events are the atomic persistence unit for model lineage.  Each event records
a single lifecycle moment (checkpoint save, promotion, match result, training
start/resume) and is serialized as one JSONL line.

Schema versioning follows the same semver-like policy as view_contracts.py:
    - Patch: optional field additions only.
    - Minor/major: required-field or semantic changes.

All types use TypedDict for zero-overhead JSON round-tripping.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypedDict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINEAGE_SCHEMA_VERSION: str = "v1.0.0"
"""Current lineage event schema version (semver-like)."""

EVENT_TYPES: tuple[str, ...] = (
    "checkpoint_created",
    "model_promoted",
    "match_completed",
    "training_started",
    "training_resumed",
)
"""Canonical event type names — order is for documentation, not semantic."""

EventType = Literal[
    "checkpoint_created",
    "model_promoted",
    "match_completed",
    "training_started",
    "training_resumed",
]
"""Allowed values for the ``event_type`` field."""


# ---------------------------------------------------------------------------
# Typed payloads (one per event type)
# ---------------------------------------------------------------------------


class CheckpointCreatedPayload(TypedDict):
    """Payload emitted when a model checkpoint is saved."""

    checkpoint_path: str
    global_timestep: int
    total_episodes: int
    parent_model_id: Optional[str]


class ModelPromotedPayload(TypedDict):
    """Payload emitted when a model is promoted (e.g. new best)."""

    from_rating: float
    to_rating: float
    promotion_reason: str


class MatchCompletedPayload(TypedDict):
    """Payload emitted after an evaluation match set completes."""

    opponent_model_id: str
    result: str  # "win", "loss", "draw", or aggregate description
    num_games: int
    win_rate: float
    agent_rating: float
    opponent_rating: float


class TrainingStartedPayload(TypedDict):
    """Payload emitted when a fresh training run begins."""

    config_snapshot: Dict[str, Any]
    parent_model_id: Optional[str]


class TrainingResumedPayload(TypedDict):
    """Payload emitted when training resumes from a checkpoint."""

    resumed_from_checkpoint: str
    global_timestep_at_resume: int
    parent_model_id: Optional[str]


# ---------------------------------------------------------------------------
# Top-level event envelope
# ---------------------------------------------------------------------------


class LineageEvent(TypedDict):
    """A single lineage event — one JSONL line in the event log."""

    event_id: str
    event_type: str  # EventType at runtime, str for forward compat
    schema_version: str
    emitted_at: str  # ISO-8601 UTC timestamp
    run_name: str
    model_id: str
    payload: Dict[str, Any]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_event_id(seq: int) -> str:
    """Build a sortable, unique event ID.

    Format: ``"{seq:06d}_{iso_utc}_{uuid8}"``

    The zero-padded sequence number ensures lexicographic sort matches
    chronological order.  The UTC timestamp provides human readability and
    the UUID suffix guarantees uniqueness across concurrent writers.
    """
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{seq:06d}_{now}_{suffix}"


def make_model_id(run_name: str, timestep: int) -> str:
    """Build a deterministic model identifier.

    Format: ``"{run_name}::checkpoint_ts{timestep}"``
    """
    return f"{run_name}::checkpoint_ts{timestep}"


def make_event(
    *,
    seq: int,
    event_type: str,
    run_name: str,
    model_id: str,
    payload: Dict[str, Any],
) -> LineageEvent:
    """Construct a complete ``LineageEvent`` with auto-generated metadata.

    Parameters
    ----------
    seq:
        Monotonic sequence number for event ID generation.
    event_type:
        One of :data:`EVENT_TYPES`.
    run_name:
        Human-readable training run identifier.
    model_id:
        Model identifier (typically from :func:`make_model_id`).
    payload:
        Event-specific payload dict.

    Returns
    -------
    LineageEvent
        Ready to serialize via ``json.dumps``.
    """
    return LineageEvent(
        event_id=make_event_id(seq),
        event_type=event_type,
        schema_version=LINEAGE_SCHEMA_VERSION,
        emitted_at=datetime.now(timezone.utc).isoformat(),
        run_name=run_name,
        model_id=model_id,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_EVENT_KEYS = {
    "event_id",
    "event_type",
    "schema_version",
    "emitted_at",
    "run_name",
    "model_id",
    "payload",
}


def validate_event(data: Dict[str, Any]) -> List[str]:
    """Check *data* against the ``LineageEvent`` contract.

    Returns a list of human-readable error strings.  An empty list means
    the event is valid.  This mirrors ``validate_envelope()`` in
    ``view_contracts.py`` — it never raises.
    """
    errors: List[str] = []

    # Required keys
    for key in _REQUIRED_EVENT_KEYS:
        if key not in data:
            errors.append(f"missing required key: {key!r}")

    if errors:
        return errors  # can't check further without required keys

    # event_id must be a non-empty string
    eid = data["event_id"]
    if not isinstance(eid, str) or not eid:
        errors.append(f"event_id must be a non-empty string, got {eid!r}")

    # event_type must be a known type
    et = data["event_type"]
    if not isinstance(et, str):
        errors.append(f"event_type must be a string, got {type(et).__name__}")
    elif et not in EVENT_TYPES:
        errors.append(
            f"event_type {et!r} is not a recognised type; "
            f"expected one of {EVENT_TYPES}"
        )

    # schema_version must be a non-empty string
    sv = data["schema_version"]
    if not isinstance(sv, str) or not sv:
        errors.append(f"schema_version must be a non-empty string, got {sv!r}")

    # emitted_at must be a non-empty string
    ea = data["emitted_at"]
    if not isinstance(ea, str) or not ea:
        errors.append(f"emitted_at must be a non-empty string, got {ea!r}")

    # run_name must be a string
    rn = data["run_name"]
    if not isinstance(rn, str):
        errors.append(f"run_name must be a string, got {type(rn).__name__}")

    # model_id must be a string
    mi = data["model_id"]
    if not isinstance(mi, str):
        errors.append(f"model_id must be a string, got {type(mi).__name__}")

    # payload must be a dict
    pl = data["payload"]
    if not isinstance(pl, dict):
        errors.append(f"payload must be a dict, got {type(pl).__name__}")

    return errors
