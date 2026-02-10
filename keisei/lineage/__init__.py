"""
keisei.lineage — Event-backed model lineage persistence.

Public API re-exports for the lineage subsystem.  Import from here
rather than reaching into sub-modules directly.
"""

from keisei.lineage.event_schema import (
    EVENT_TYPES,
    LINEAGE_SCHEMA_VERSION,
    CheckpointCreatedPayload,
    EventType,
    LineageEvent,
    MatchCompletedPayload,
    ModelPromotedPayload,
    TrainingResumedPayload,
    TrainingStartedPayload,
    make_event,
    make_event_id,
    make_model_id,
    validate_event,
)
from keisei.lineage.graph import LineageGraph, MatchRecord, ModelNode, PromotionRecord
from keisei.lineage.registry import LineageRegistry

__all__ = [
    "LINEAGE_SCHEMA_VERSION",
    "EVENT_TYPES",
    "EventType",
    "LineageEvent",
    "CheckpointCreatedPayload",
    "ModelPromotedPayload",
    "MatchCompletedPayload",
    "TrainingStartedPayload",
    "TrainingResumedPayload",
    "make_event_id",
    "make_event",
    "make_model_id",
    "validate_event",
    "LineageRegistry",
    "LineageGraph",
    "ModelNode",
    "MatchRecord",
    "PromotionRecord",
]
