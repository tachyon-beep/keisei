"""Tests for game_features and style_profiles DB tables."""

import json
import tempfile

import pytest

from keisei.db import (
    init_db,
    read_all_game_features,
    read_game_features_for_checkpoint,
    read_style_profiles,
    write_game_features,
    write_style_profile,
)


@pytest.fixture
def db_path():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    init_db(f.name)
    # Insert league entries for FK constraints
    import sqlite3
    conn = sqlite3.connect(f.name)
    conn.execute(
        "INSERT INTO league_entries "
        "(id, display_name, flavour_facts, architecture, model_params, "
        "checkpoint_path, elo_rating, created_epoch) "
        "VALUES (1, 'A', '[]', 'resnet', '{}', '/a', 1000, 0)"
    )
    conn.execute(
        "INSERT INTO league_entries "
        "(id, display_name, flavour_facts, architecture, model_params, "
        "checkpoint_path, elo_rating, created_epoch) "
        "VALUES (2, 'B', '[]', 'resnet', '{}', '/b', 1000, 0)"
    )
    conn.commit()
    conn.close()
    return f.name


class TestGameFeatures:
    def test_write_and_read(self, db_path):
        features = [{
            "checkpoint_id": 1,
            "opponent_id": 2,
            "epoch": 5,
            "side": "black",
            "result": "win",
            "total_plies": 80,
            "first_action": 100,
            "opening_seq_3": "100,200,300",
            "opening_seq_6": None,
            "rook_moved_ply": 15,
            "king_displacement_20": 2,
            "first_capture_ply": 20,
            "first_check_ply": None,
            "first_drop_ply": 30,
            "num_checks": 0,
            "num_captures": 5,
            "num_drops": 3,
            "num_promotions": 2,
            "num_early_drops": 1,
            "rook_moves_in_20": 2,
            "king_moves_in_30": 1,
            "num_repetitions": 0,
            "termination_reason": 1,
        }]
        write_game_features(db_path, features)
        rows = read_game_features_for_checkpoint(db_path, 1)
        assert len(rows) == 1
        assert rows[0]["side"] == "black"
        assert rows[0]["result"] == "win"
        assert rows[0]["total_plies"] == 80
        assert rows[0]["num_drops"] == 3
        assert rows[0]["first_capture_ply"] == 20

    def test_write_empty_list(self, db_path):
        write_game_features(db_path, [])
        rows = read_all_game_features(db_path)
        assert len(rows) == 0

    def test_read_all(self, db_path):
        for cid in [1, 2]:
            write_game_features(db_path, [{
                "checkpoint_id": cid,
                "opponent_id": 2 if cid == 1 else 1,
                "epoch": 5,
                "side": "black",
                "result": "win",
                "total_plies": 80,
                "num_drops": 0,
                "num_promotions": 0,
                "num_captures": 0,
                "num_early_drops": 0,
                "rook_moves_in_20": 0,
                "king_moves_in_30": 0,
                "num_repetitions": 0,
                "num_checks": 0,
                "king_displacement_20": 0,
                "termination_reason": 0,
            }])
        rows = read_all_game_features(db_path)
        assert len(rows) == 2

    def test_null_optional_fields(self, db_path):
        features = [{
            "checkpoint_id": 1,
            "opponent_id": 2,
            "epoch": 5,
            "side": "white",
            "result": "loss",
            "total_plies": 60,
            "num_drops": 0,
            "num_promotions": 0,
            "num_captures": 0,
            "num_early_drops": 0,
            "rook_moves_in_20": 0,
            "king_moves_in_30": 0,
            "num_repetitions": 0,
            "num_checks": 0,
            "king_displacement_20": 0,
            "termination_reason": 0,
            # These are optional (NULL)
            "first_action": None,
            "opening_seq_3": None,
            "opening_seq_6": None,
            "rook_moved_ply": None,
            "first_capture_ply": None,
            "first_check_ply": None,
            "first_drop_ply": None,
        }]
        write_game_features(db_path, features)
        rows = read_game_features_for_checkpoint(db_path, 1)
        assert rows[0]["first_action"] is None
        assert rows[0]["first_capture_ply"] is None


class TestStyleProfiles:
    def test_write_and_read(self, db_path):
        write_style_profile(db_path, {
            "checkpoint_id": 1,
            "recomputed_at": "2026-01-01T00:00:00Z",
            "profile_status": "established",
            "games_sampled": 100,
            "raw_metrics": {"avg_game_length": 80.5, "drops_per_game": 2.1},
            "percentiles": {"avg_game_length": 55.0, "drops_per_game": 72.0},
            "primary_style": "Patient attacker",
            "secondary_traits": ["Long games", "High captures"],
            "commentary": [
                {"text": "Longer games than most", "category": "game_length", "confidence": "high"}
            ],
        })
        profiles = read_style_profiles(db_path)
        assert len(profiles) == 1
        p = profiles[0]
        assert p["checkpoint_id"] == 1
        assert p["profile_status"] == "established"
        assert p["primary_style"] == "Patient attacker"
        assert p["secondary_traits"] == ["Long games", "High captures"]
        assert p["raw_metrics"]["avg_game_length"] == 80.5
        assert p["commentary"][0]["text"] == "Longer games than most"

    def test_upsert(self, db_path):
        write_style_profile(db_path, {
            "checkpoint_id": 1,
            "recomputed_at": "2026-01-01T00:00:00Z",
            "profile_status": "provisional",
            "games_sampled": 50,
            "raw_metrics": {},
            "percentiles": {},
            "primary_style": None,
            "secondary_traits": [],
            "commentary": [],
        })
        write_style_profile(db_path, {
            "checkpoint_id": 1,
            "recomputed_at": "2026-01-02T00:00:00Z",
            "profile_status": "established",
            "games_sampled": 100,
            "raw_metrics": {"test": 1},
            "percentiles": {},
            "primary_style": "Slow builder",
            "secondary_traits": ["Foo"],
            "commentary": [],
        })
        profiles = read_style_profiles(db_path)
        assert len(profiles) == 1
        assert profiles[0]["profile_status"] == "established"
        assert profiles[0]["primary_style"] == "Slow builder"
        assert profiles[0]["games_sampled"] == 100


class TestSchemaVersion:
    def test_init_creates_tables(self, db_path):
        """Ensure both new tables exist after init_db."""
        import sqlite3
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "game_features" in tables
        assert "style_profiles" in tables

    def test_version_is_2(self, db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        conn.close()
        assert version == 2
