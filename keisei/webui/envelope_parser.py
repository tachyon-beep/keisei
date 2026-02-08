"""Read-only access layer for BroadcastStateEnvelope payloads.

Sits between the raw JSON dict (loaded from the state file) and the
Streamlit rendering functions.  The renderer never accesses envelope keys
directly — it reads through this parser.

This module deliberately avoids heavy imports so that the Streamlit process
does not pull in torch/cuda/training code.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .view_contracts import STALE_THRESHOLD_SECONDS, VIEW_KEYS


class EnvelopeParser:
    """Typed read-only accessor for a v1 BroadcastStateEnvelope.

    Provides convenience properties for the training view sub-keys and
    envelope-level metadata.  Returns ``None`` / empty defaults when
    data is absent — rendering code never needs to guard against KeyError.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: Dict[str, Any]) -> None:
        self._raw = raw

    # -- envelope metadata --------------------------------------------------

    @property
    def schema_version(self) -> str:
        return self._raw.get("schema_version", "")

    @property
    def timestamp(self) -> float:
        return self._raw.get("timestamp", 0.0)

    @property
    def speed(self) -> float:
        return self._raw.get("speed", 0.0)

    @property
    def mode(self) -> str:
        return self._raw.get("mode", "training_only")

    @property
    def active_views(self) -> List[str]:
        return self._raw.get("active_views", [])

    @property
    def health(self) -> Dict[str, str]:
        return self._raw.get("health", {})

    @property
    def pending_updates(self) -> Dict[str, Any]:
        return self._raw.get("pending_updates", {})

    # -- training view sub-keys ---------------------------------------------

    @property
    def training(self) -> Optional[Dict[str, Any]]:
        return self._raw.get("training")

    @property
    def board_state(self) -> Optional[Dict[str, Any]]:
        t = self.training
        return t.get("board_state") if t else None

    @property
    def metrics(self) -> Dict[str, Any]:
        t = self.training
        return t.get("metrics", {}) if t else {}

    @property
    def step_info(self) -> Optional[Dict[str, Any]]:
        t = self.training
        return t.get("step_info") if t else None

    @property
    def buffer_info(self) -> Optional[Dict[str, Any]]:
        t = self.training
        return t.get("buffer_info") if t else None

    @property
    def model_info(self) -> Dict[str, Any]:
        t = self.training
        return t.get("model_info", {}) if t else {}

    # -- health / staleness -------------------------------------------------

    def view_health(self, view: str) -> str:
        """Health status for *view*, defaulting to ``'missing'``."""
        h = self.health
        if not isinstance(h, dict):
            return "missing"
        return h.get(view, "missing")

    def is_stale(self, threshold: float = STALE_THRESHOLD_SECONDS) -> bool:
        """True when the snapshot is older than *threshold* seconds."""
        ts = self.timestamp
        if ts <= 0:
            return True
        return (time.time() - ts) > threshold

    def age_seconds(self) -> float:
        """Seconds since the snapshot was created."""
        ts = self.timestamp
        if ts <= 0:
            return float("inf")
        return time.time() - ts

    def has_view(self, view: str) -> bool:
        """True when *view* is listed in ``active_views``."""
        return view in self.active_views

    def available_optional_views(self) -> List[str]:
        """Optional view keys that are active (not just training)."""
        return [v for v in self.active_views if v != "training"]

    def missing_optional_views(self) -> List[str]:
        """Optional view keys that are NOT active."""
        return [v for v in VIEW_KEYS if v != "training" and v not in self.active_views]
