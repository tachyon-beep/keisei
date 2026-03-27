"""
Tests for ladder evaluator integration with the real EloTracker and EloRegistry.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestLadderUsesRealEloTracker:
    """Verify the import in ladder.py is the real EloTracker, not a placeholder."""

    def test_ladder_elo_tracker_is_real_import(self):
        """The EloTracker used by LadderEvaluator should be from analytics.elo_tracker."""
        from keisei.evaluation.strategies.ladder import EloTracker as LadderEloTracker
        from keisei.evaluation.analytics.elo_tracker import EloTracker as RealEloTracker

        assert LadderEloTracker is RealEloTracker, (
            "LadderEvaluator should use the real EloTracker from analytics, not a placeholder"
        )

    def test_ladder_evaluator_creates_real_elo_tracker(self):
        """LadderEvaluator.__init__ should instantiate the real EloTracker."""
        from keisei.evaluation.analytics.elo_tracker import EloTracker as RealEloTracker
        from keisei.evaluation.strategies.ladder import LadderEvaluator

        config = MagicMock()
        config.get_strategy_param = MagicMock(return_value={})
        config.strategy_params = {}
        config.log_level = "INFO"

        evaluator = LadderEvaluator(config)
        assert isinstance(evaluator.elo_tracker, RealEloTracker)


class TestLadderCallSiteCompatibility:
    """Verify the real EloTracker API methods work as called by LadderEvaluator."""

    def test_get_rating_auto_creates(self):
        """get_rating should auto-create an entity with default rating."""
        from keisei.evaluation.analytics.elo_tracker import EloTracker

        tracker = EloTracker()
        rating = tracker.get_rating("new_agent")
        assert rating == 1500.0
        assert "new_agent" in tracker.ratings

    def test_update_rating_per_game(self):
        """update_rating should update ratings for a single game outcome."""
        from keisei.evaluation.analytics.elo_tracker import EloTracker

        tracker = EloTracker()
        new_a, new_b = tracker.update_rating("agent", "opponent", 1.0)
        # Agent won, so agent rating should increase
        assert new_a > 1500.0
        assert new_b < 1500.0

    def test_get_all_ratings_returns_copy(self):
        """get_all_ratings should return a dict copy of all ratings."""
        from keisei.evaluation.analytics.elo_tracker import EloTracker

        tracker = EloTracker(initial_ratings={"a": 1600.0, "b": 1400.0})
        snapshot = tracker.get_all_ratings()
        assert snapshot == {"a": 1600.0, "b": 1400.0}
        # Verify it's a copy
        snapshot["a"] = 9999.0
        assert tracker.ratings["a"] == 1600.0

    def test_default_initial_rating_attribute(self):
        """EloTracker should have default_initial_rating attribute."""
        from keisei.evaluation.analytics.elo_tracker import EloTracker

        tracker = EloTracker()
        assert tracker.default_initial_rating == 1500.0

        tracker2 = EloTracker(default_initial_rating=1200.0)
        assert tracker2.default_initial_rating == 1200.0


class TestLadderEloPersistence:
    """Verify EloRegistry round-trip persistence works for ladder integration."""

    def test_elo_registry_save_and_load_roundtrip(self):
        """EloRegistry should save ratings and load them back correctly."""
        from keisei.evaluation.opponents.elo_registry import EloRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "elo_ratings.json"

            # Save
            registry = EloRegistry(path)
            registry.ratings = {"agent_1": 1550.0, "opponent_1": 1450.0}
            registry.save()

            # Verify file exists
            assert path.exists()

            # Load in a new instance
            registry2 = EloRegistry(path)
            assert registry2.ratings == {"agent_1": 1550.0, "opponent_1": 1450.0}

    def test_elo_registry_seeds_elo_tracker(self):
        """EloRegistry ratings should seed EloTracker via initial_ratings."""
        from keisei.evaluation.analytics.elo_tracker import EloTracker
        from keisei.evaluation.opponents.elo_registry import EloRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "elo_ratings.json"

            # Create registry with some ratings
            registry = EloRegistry(path)
            registry.ratings = {"player_a": 1600.0, "player_b": 1400.0}
            registry.save()

            # Load and seed tracker
            registry2 = EloRegistry(path)
            tracker = EloTracker(initial_ratings=dict(registry2.ratings))

            assert tracker.get_rating("player_a") == 1600.0
            assert tracker.get_rating("player_b") == 1400.0

    def test_elo_tracker_updates_save_back_to_registry(self):
        """After EloTracker updates, ratings should save back via EloRegistry."""
        from keisei.evaluation.analytics.elo_tracker import EloTracker
        from keisei.evaluation.opponents.elo_registry import EloRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "elo_ratings.json"

            # Initial state
            registry = EloRegistry(path)
            registry.ratings = {"agent": 1500.0, "opp": 1500.0}
            registry.save()

            # Simulate: load, create tracker, update, save back
            registry2 = EloRegistry(path)
            tracker = EloTracker(initial_ratings=dict(registry2.ratings))
            tracker.update_rating("agent", "opp", 1.0)  # agent wins

            # Save back
            registry2.ratings = tracker.get_all_ratings()
            registry2.save()

            # Verify persistence
            registry3 = EloRegistry(path)
            assert registry3.ratings["agent"] > 1500.0
            assert registry3.ratings["opp"] < 1500.0
