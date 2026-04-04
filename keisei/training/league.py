"""Backward-compatibility shim -- will be removed in the final cleanup task."""
from keisei.training.opponent_store import (  # noqa: F401
    OpponentEntry,
    OpponentStore as OpponentPool,
    Role,
    EntryStatus,
    compute_elo_update,
)

# OpponentSampler is removed -- use MatchScheduler instead.
