"""Historical library table — fixed-slot opponent pool for gauntlets."""

from __future__ import annotations

DDL = """
CREATE TABLE IF NOT EXISTS historical_library (
    slot_index     INTEGER NOT NULL PRIMARY KEY,
    target_epoch   INTEGER NOT NULL,
    entry_id       INTEGER REFERENCES league_entries(id),
    actual_epoch   INTEGER,
    selected_at    TEXT NOT NULL,
    selection_mode TEXT NOT NULL
);
"""
