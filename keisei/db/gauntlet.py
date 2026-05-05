"""Gauntlet results table — periodic ladder evaluations against historical pool."""

from __future__ import annotations

DDL = """
CREATE TABLE IF NOT EXISTS gauntlet_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch               INTEGER NOT NULL,
    entry_id            INTEGER NOT NULL REFERENCES league_entries(id),
    historical_slot     INTEGER NOT NULL,
    historical_entry_id INTEGER NOT NULL REFERENCES league_entries(id),
    wins                INTEGER NOT NULL,
    losses              INTEGER NOT NULL,
    draws               INTEGER NOT NULL,
    elo_before          REAL,
    elo_after           REAL,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_gauntlet_epoch ON gauntlet_results(epoch);
"""
