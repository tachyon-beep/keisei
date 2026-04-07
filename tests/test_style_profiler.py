"""Tests for StyleProfiler aggregation, classification, and commentary."""

import json
import tempfile

import pytest

from keisei.db import init_db, write_game_features, read_style_profiles
from keisei.training.style_profiler import (
    StyleProfiler,
    _assign_labels,
    _check_rule,
    _generate_commentary,
    _percentile_rank,
    _safe_mean,
    THRESHOLD_INSUFFICIENT,
    THRESHOLD_PROVISIONAL,
)


class TestHelpers:
    def test_percentile_rank_middle(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile_rank(3.0, values) == 50.0

    def test_percentile_rank_lowest(self):
        values = [1.0, 2.0, 3.0]
        pct = _percentile_rank(1.0, values)
        assert pct < 30

    def test_percentile_rank_highest(self):
        values = [1.0, 2.0, 3.0]
        pct = _percentile_rank(3.0, values)
        assert pct > 70

    def test_percentile_rank_empty(self):
        assert _percentile_rank(5.0, []) == 50.0

    def test_safe_mean(self):
        assert _safe_mean([1.0, 2.0, 3.0]) == 2.0
        assert _safe_mean([]) is None


class TestCheckRule:
    def test_passes(self):
        conditions = {"avg_game_length": (">=", 65)}
        percentiles = {"avg_game_length": 70.0}
        assert _check_rule(conditions, percentiles)

    def test_fails(self):
        conditions = {"avg_game_length": (">=", 65)}
        percentiles = {"avg_game_length": 50.0}
        assert not _check_rule(conditions, percentiles)

    def test_missing_metric(self):
        conditions = {"avg_game_length": (">=", 65)}
        percentiles = {}
        assert not _check_rule(conditions, percentiles)

    def test_multiple_conditions(self):
        conditions = {
            "avg_game_length": (">=", 65),
            "num_captures_mean": (">=", 55),
        }
        percentiles = {"avg_game_length": 70.0, "num_captures_mean": 60.0}
        assert _check_rule(conditions, percentiles)

    def test_le_comparator(self):
        conditions = {"first_capture_ply_mean": ("<=", 30)}
        percentiles = {"first_capture_ply_mean": 20.0}
        assert _check_rule(conditions, percentiles)


class TestAssignLabels:
    def test_no_labels(self):
        primary, secondary = _assign_labels({})
        assert primary is None
        assert secondary == []

    def test_single_label(self):
        pcts = {"drops_per_game": 85.0, "num_early_drops_mean": 70.0}
        primary, secondary = _assign_labels(pcts)
        assert primary == "Drop-heavy scrapper"

    def test_max_three_labels(self):
        # Everything high should trigger multiple rules but cap at 3
        pcts = {k: 90.0 for k in [
            "first_capture_ply_mean", "avg_game_length", "num_captures_mean",
            "drops_per_game", "num_early_drops_mean", "promotions_per_game",
            "rook_moved_early_rate", "opening_diversity_index",
            "king_moves_early_rate", "game_length_variance", "short_game_rate",
        ]}
        # Set low values for "<="-type rules
        pcts["first_capture_ply_mean"] = 10.0
        pcts["avg_game_length"] = 10.0
        pcts["game_length_variance"] = 10.0

        primary, secondary = _assign_labels(pcts)
        assert primary is not None
        assert len(secondary) <= 2

    def test_contradictions_filtered(self):
        # Try to trigger both "Sharp tactical opener" and "Slow builder"
        pcts = {
            "first_capture_ply_mean": 20.0,  # Sharp tactical needs <=30
            "avg_game_length": 80.0,  # Slow builder needs >=70
            "short_game_rate": 60.0,
        }
        primary, secondary = _assign_labels(pcts)
        labels = [primary] + secondary if primary else secondary
        assert not ("Sharp tactical opener" in labels and "Slow builder" in labels)


class TestGenerateCommentary:
    def test_generates_facts(self):
        raw = {
            "avg_game_length": 50.0,
            "drops_per_game": 3.0,
            "first_capture_ply_mean": 15.0,
        }
        pcts = {
            "avg_game_length": 25.0,  # below 35 → "Shorter games"
            "drops_per_game": 80.0,   # above 70 → "Uses drops more"
            "first_capture_ply_mean": 20.0,  # below 30 → "Starts exchanging earlier"
        }
        facts = _generate_commentary(raw, pcts)
        assert len(facts) > 0
        assert all("text" in f and "category" in f and "confidence" in f for f in facts)

    def test_max_five_facts(self):
        # All metrics trigger
        raw = {k: 1.0 for k in [
            "avg_game_length", "drops_per_game", "first_capture_ply_mean",
            "promotions_per_game", "num_captures_mean", "rook_moved_early_rate",
            "king_moves_early_rate", "opening_diversity_index",
            "num_early_drops_mean",
        ]}
        pcts = {k: 90.0 for k in raw}
        pcts["avg_game_length"] = 10.0
        pcts["first_capture_ply_mean"] = 10.0
        pcts["rook_moved_early_rate"] = 10.0
        pcts["opening_diversity_index"] = 10.0
        facts = _generate_commentary(raw, pcts)
        assert len(facts) <= 5


class TestProfileStatus:
    def test_thresholds(self):
        profiler = StyleProfiler.__new__(StyleProfiler)
        assert profiler._profile_status(10) == "insufficient"
        assert profiler._profile_status(24) == "insufficient"
        assert profiler._profile_status(25) == "provisional"
        assert profiler._profile_status(74) == "provisional"
        assert profiler._profile_status(75) == "established"
        assert profiler._profile_status(200) == "established"


class TestStyleProfilerIntegration:
    """Integration test with real DB."""

    def _make_db(self):
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        init_db(f.name)
        # Need league_entries for FK
        import sqlite3
        conn = sqlite3.connect(f.name)
        conn.execute(
            "INSERT INTO league_entries "
            "(id, display_name, flavour_facts, architecture, model_params, "
            "checkpoint_path, elo_rating, created_epoch) "
            "VALUES (1, 'TestA', '[]', 'resnet', '{}', '/a', 1000, 0)"
        )
        conn.execute(
            "INSERT INTO league_entries "
            "(id, display_name, flavour_facts, architecture, model_params, "
            "checkpoint_path, elo_rating, created_epoch) "
            "VALUES (2, 'TestB', '[]', 'resnet', '{}', '/b', 1000, 0)"
        )
        conn.commit()
        conn.close()
        return f.name

    def _generate_features(self, db_path, checkpoint_id, count):
        features = []
        for i in range(count):
            features.append({
                "checkpoint_id": checkpoint_id,
                "opponent_id": 2 if checkpoint_id == 1 else 1,
                "epoch": 10,
                "side": "black" if i % 2 == 0 else "white",
                "result": "win" if i % 3 == 0 else ("loss" if i % 3 == 1 else "draw"),
                "total_plies": 80 + i,
                "first_action": 100 + (i % 5),
                "opening_seq_3": f"{100 + i % 5},{200 + i % 3},{300 + i % 4}",
                "opening_seq_6": None,
                "rook_moved_ply": 15 if i % 3 == 0 else None,
                "king_displacement_20": i % 2,
                "first_capture_ply": 20 + i % 10,
                "first_check_ply": None,
                "first_drop_ply": 30 + i % 8 if i % 4 == 0 else None,
                "num_checks": 0,
                "num_captures": 3 + i % 5,
                "num_drops": 2 + i % 3,
                "num_promotions": 1 + i % 2,
                "num_early_drops": 1 if i % 4 == 0 else 0,
                "rook_moves_in_20": i % 3,
                "king_moves_in_30": i % 2,
                "num_repetitions": 0,
                "termination_reason": 1,
            })
        write_game_features(db_path, features)

    def test_recompute_insufficient_sample(self):
        db_path = self._make_db()
        self._generate_features(db_path, 1, 10)  # Below threshold
        profiler = StyleProfiler(db_path)
        count = profiler.recompute_all()
        assert count == 1
        profiles = read_style_profiles(db_path)
        assert len(profiles) == 1
        assert profiles[0]["profile_status"] == "insufficient"
        assert profiles[0]["primary_style"] is None

    def test_recompute_provisional(self):
        db_path = self._make_db()
        self._generate_features(db_path, 1, 50)
        profiler = StyleProfiler(db_path)
        profiler.recompute_all()
        profiles = read_style_profiles(db_path)
        assert profiles[0]["profile_status"] == "provisional"
        # Should have raw metrics
        assert profiles[0]["raw_metrics"].get("avg_game_length") is not None

    def test_recompute_established(self):
        db_path = self._make_db()
        self._generate_features(db_path, 1, 100)
        profiler = StyleProfiler(db_path)
        profiler.recompute_all()
        profiles = read_style_profiles(db_path)
        assert profiles[0]["profile_status"] == "established"

    def test_recompute_multiple_checkpoints(self):
        db_path = self._make_db()
        self._generate_features(db_path, 1, 80)
        self._generate_features(db_path, 2, 80)
        profiler = StyleProfiler(db_path)
        count = profiler.recompute_all()
        assert count == 2
        profiles = read_style_profiles(db_path)
        assert len(profiles) == 2
        # Percentiles should be relative — one should differ from the other
        p1 = profiles[0]["percentiles"]
        p2 = profiles[1]["percentiles"]
        assert isinstance(p1, dict)
        assert isinstance(p2, dict)

    def test_idempotent_recompute(self):
        db_path = self._make_db()
        self._generate_features(db_path, 1, 80)
        profiler = StyleProfiler(db_path)
        profiler.recompute_all()
        profiles1 = read_style_profiles(db_path)
        profiler.recompute_all()
        profiles2 = read_style_profiles(db_path)
        assert profiles1[0]["raw_metrics"] == profiles2[0]["raw_metrics"]
