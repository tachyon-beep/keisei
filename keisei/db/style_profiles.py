"""Style profiles table — per-checkpoint percentile/style summaries."""

from __future__ import annotations

import json
from typing import Any

from keisei.db._connection import _connect

DDL = """
CREATE TABLE IF NOT EXISTS style_profiles (
    checkpoint_id       INTEGER PRIMARY KEY REFERENCES league_entries(id),
    recomputed_at       TEXT NOT NULL,
    profile_status      TEXT NOT NULL DEFAULT 'insufficient',
    games_sampled       INTEGER NOT NULL DEFAULT 0,
    raw_metrics_json    TEXT NOT NULL DEFAULT '{}',
    percentile_json     TEXT NOT NULL DEFAULT '{}',
    primary_style       TEXT,
    secondary_traits    TEXT NOT NULL DEFAULT '[]',
    commentary_json     TEXT NOT NULL DEFAULT '[]',
    updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""


def write_style_profile(db_path: str, profile: dict[str, Any]) -> None:
    """Upsert a single checkpoint style profile."""
    conn = _connect(db_path)
    try:
        conn.execute(
            """INSERT INTO style_profiles
               (checkpoint_id, recomputed_at, profile_status, games_sampled,
                raw_metrics_json, percentile_json, primary_style,
                secondary_traits, commentary_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                       strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
               ON CONFLICT(checkpoint_id) DO UPDATE SET
                 recomputed_at    = excluded.recomputed_at,
                 profile_status   = excluded.profile_status,
                 games_sampled    = excluded.games_sampled,
                 raw_metrics_json = excluded.raw_metrics_json,
                 percentile_json  = excluded.percentile_json,
                 primary_style    = excluded.primary_style,
                 secondary_traits = excluded.secondary_traits,
                 commentary_json  = excluded.commentary_json,
                 updated_at       = excluded.updated_at""",
            (
                profile["checkpoint_id"],
                profile["recomputed_at"],
                profile["profile_status"],
                profile["games_sampled"],
                json.dumps(profile.get("raw_metrics", {})),
                json.dumps(profile.get("percentiles", {})),
                profile.get("primary_style"),
                json.dumps(profile.get("secondary_traits", [])),
                json.dumps(profile.get("commentary", [])),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def read_style_profiles(db_path: str) -> list[dict[str, Any]]:
    """Read all style profiles for the league."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM style_profiles ORDER BY checkpoint_id"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["raw_metrics"] = json.loads(d.pop("raw_metrics_json"))
            d["percentiles"] = json.loads(d.pop("percentile_json"))
            d["secondary_traits"] = json.loads(d["secondary_traits"])
            d["commentary"] = json.loads(d.pop("commentary_json"))
            result.append(d)
        return result
    finally:
        conn.close()
