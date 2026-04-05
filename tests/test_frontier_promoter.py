"""Tests for FrontierPromoter — multi-criteria promotion from Dynamic to Frontier Static."""

from __future__ import annotations

import pytest

from keisei.config import FrontierStaticConfig
from keisei.training.frontier_promoter import FrontierPromoter
from keisei.training.opponent_store import OpponentEntry, Role


def _make_entry(
    id: int,
    elo: float = 1000.0,
    games: int = 0,
    role: Role = Role.DYNAMIC,
    lineage: str | None = None,
) -> OpponentEntry:
    """Helper to build minimal OpponentEntry for promoter tests."""
    return OpponentEntry(
        id=id,
        display_name=f"test-{id}",
        architecture="resnet",
        model_params={},
        checkpoint_path="/tmp/test.pt",
        elo_rating=elo,
        created_epoch=0,
        games_played=games,
        created_at="2026-01-01T00:00:00Z",
        flavour_facts=[],
        role=role,
        lineage_group=lineage,
    )


@pytest.fixture
def config() -> FrontierStaticConfig:
    return FrontierStaticConfig(
        slots=5,
        review_interval_epochs=250,
        min_tenure_epochs=100,
        promotion_margin_elo=50.0,
        min_games_for_promotion=100,
        topk=3,
        streak_epochs=50,
        max_lineage_overlap=2,
    )


class TestEvaluate:
    def test_evaluate_returns_none_when_no_candidates(self, config: FrontierStaticConfig) -> None:
        """3 Dynamic entries with games < min_games_for_promotion -> None."""
        promoter = FrontierPromoter(config)
        dynamics = [
            _make_entry(1, elo=1300, games=10),
            _make_entry(2, elo=1250, games=20),
            _make_entry(3, elo=1200, games=30),
        ]
        frontier = [_make_entry(10, elo=1100, role=Role.FRONTIER_STATIC)]

        # Run enough epochs for streak
        for epoch in range(60):
            result = promoter.evaluate(dynamics, frontier, epoch)

        assert result is None

    def test_evaluate_returns_none_below_elo_margin(self, config: FrontierStaticConfig) -> None:
        """Dynamic Elo only 20 above weakest Frontier (below margin of 50) -> None."""
        promoter = FrontierPromoter(config)
        dynamics = [
            _make_entry(1, elo=1120, games=200),  # only 20 above weakest frontier
            _make_entry(2, elo=1110, games=200),
            _make_entry(3, elo=1105, games=200),
        ]
        frontier = [
            _make_entry(10, elo=1200, role=Role.FRONTIER_STATIC),
            _make_entry(11, elo=1100, role=Role.FRONTIER_STATIC),  # weakest
        ]

        for epoch in range(60):
            result = promoter.evaluate(dynamics, frontier, epoch)

        assert result is None

    def test_evaluate_returns_candidate_when_all_criteria_met(
        self, config: FrontierStaticConfig
    ) -> None:
        """Top Dynamic (Elo 1250, games=100) with streak -> returns candidate."""
        promoter = FrontierPromoter(config)
        dynamics = [
            _make_entry(1, elo=1250, games=200, lineage="lineage-a"),
            _make_entry(2, elo=1200, games=200, lineage="lineage-b"),
            _make_entry(3, elo=1150, games=200, lineage="lineage-c"),
        ]
        frontier = [
            _make_entry(10, elo=1100, role=Role.FRONTIER_STATIC, lineage="lineage-x"),
            _make_entry(11, elo=1050, role=Role.FRONTIER_STATIC, lineage="lineage-y"),
            _make_entry(12, elo=1000, role=Role.FRONTIER_STATIC, lineage="lineage-z"),
        ]

        result = None
        for epoch in range(60):
            result = promoter.evaluate(dynamics, frontier, epoch)
            if result is not None:
                break

        assert result is not None
        assert result.id == 1

    def test_streak_tracking_across_reviews(self, config: FrontierStaticConfig) -> None:
        """Build streak, verify qualification, then drop out + reset."""
        promoter = FrontierPromoter(config)
        strong = _make_entry(1, elo=1300, games=200, lineage="lineage-a")
        # Medium and weak are below the elo margin threshold (need >= 1000 + 50)
        medium = _make_entry(2, elo=1040, games=200, lineage="lineage-b")
        weak = _make_entry(3, elo=1020, games=200, lineage="lineage-c")
        frontier = [_make_entry(10, elo=1000, role=Role.FRONTIER_STATIC)]

        # Build streak for 50 epochs
        for epoch in range(50):
            promoter.evaluate([strong, medium, weak], frontier, epoch)

        # At epoch 50, strong should qualify
        result = promoter.evaluate([strong, medium, weak], frontier, 50)
        assert result is not None
        assert result.id == 1

        # Now strong drops out of the pool — replaced by a weaker entry
        replacement = _make_entry(4, elo=1010, games=200, lineage="lineage-d")
        promoter.evaluate([medium, weak, replacement], frontier, 51)

        # Re-introduce strong — streak should be reset
        result = promoter.evaluate([strong, medium, weak], frontier, 52)
        # strong just re-entered at epoch 52, needs 50 more epochs
        # medium/weak are below elo margin, so no one qualifies
        assert result is None

    def test_streak_resets_on_fresh_promoter(self, config: FrontierStaticConfig) -> None:
        """Fresh instance has no streaks even if entry was previously top-K."""
        promoter1 = FrontierPromoter(config)
        dynamics = [_make_entry(1, elo=1300, games=200, lineage="lineage-a")]
        frontier = [_make_entry(10, elo=1000, role=Role.FRONTIER_STATIC)]

        # Build streak in promoter1
        for epoch in range(55):
            promoter1.evaluate(dynamics, frontier, epoch)

        # New promoter — no streak history
        promoter2 = FrontierPromoter(config)
        result = promoter2.evaluate(dynamics, frontier, 55)
        # Even at epoch 55, fresh promoter has no streak data
        assert result is None


class TestShouldPromote:
    def test_should_promote_checks_all_five_criteria(
        self, config: FrontierStaticConfig
    ) -> None:
        """Test each criterion independently fails, then all met -> True."""
        promoter = FrontierPromoter(config)
        frontier = [
            _make_entry(10, elo=1100, role=Role.FRONTIER_STATIC, lineage="lineage-x"),
        ]

        # Criterion 1: games_played too low
        low_games = _make_entry(1, elo=1200, games=50, lineage="lineage-a")
        promoter._topk_streaks[1] = 0
        assert promoter.should_promote(low_games, frontier, 100) is False

        # Criterion 3: streak too short
        short_streak = _make_entry(2, elo=1200, games=200, lineage="lineage-a")
        promoter._topk_streaks[2] = 80  # first seen at 80, current epoch 100 => 20 < 50
        assert promoter.should_promote(short_streak, frontier, 100) is False

        # Criterion 2: not in top-K tracking
        no_streak = _make_entry(3, elo=1200, games=200, lineage="lineage-a")
        # Don't add to _topk_streaks
        assert promoter.should_promote(no_streak, frontier, 100) is False

        # Criterion 4: elo margin too low
        low_elo = _make_entry(4, elo=1120, games=200, lineage="lineage-a")
        promoter._topk_streaks[4] = 0
        assert promoter.should_promote(low_elo, frontier, 100) is False

        # Criterion 5: lineage overlap
        overlap = _make_entry(5, elo=1200, games=200, lineage="lineage-x")
        frontier_overlap = [
            _make_entry(10, elo=1100, role=Role.FRONTIER_STATIC, lineage="lineage-x"),
            _make_entry(11, elo=1050, role=Role.FRONTIER_STATIC, lineage="lineage-x"),
        ]
        promoter._topk_streaks[5] = 0
        assert promoter.should_promote(overlap, frontier_overlap, 100) is False

        # All criteria met
        good = _make_entry(6, elo=1200, games=200, lineage="lineage-a")
        promoter._topk_streaks[6] = 0
        assert promoter.should_promote(good, frontier, 100) is True

    def test_should_promote_when_frontier_empty(
        self, config: FrontierStaticConfig
    ) -> None:
        """Empty frontier -> True (seed the tier)."""
        promoter = FrontierPromoter(config)
        candidate = _make_entry(1, elo=1000, games=0)
        assert promoter.should_promote(candidate, [], 0) is True

    def test_lineage_overlap_limit(self, config: FrontierStaticConfig) -> None:
        """2 Frontier with same lineage, Dynamic same lineage -> False. Different -> True."""
        promoter = FrontierPromoter(config)
        frontier = [
            _make_entry(10, elo=1100, role=Role.FRONTIER_STATIC, lineage="lineage-x"),
            _make_entry(11, elo=1050, role=Role.FRONTIER_STATIC, lineage="lineage-x"),
            _make_entry(12, elo=1000, role=Role.FRONTIER_STATIC, lineage="lineage-y"),
        ]

        # Same lineage as the 2 existing -> blocked (2 >= max_lineage_overlap=2)
        same_lin = _make_entry(1, elo=1200, games=200, lineage="lineage-x")
        promoter._topk_streaks[1] = 0
        assert promoter.should_promote(same_lin, frontier, 100) is False

        # Different lineage -> allowed
        diff_lin = _make_entry(2, elo=1200, games=200, lineage="lineage-z")
        promoter._topk_streaks[2] = 0
        assert promoter.should_promote(diff_lin, frontier, 100) is True

        # None lineage on candidate -> lineage check passes (no match)
        no_lin = _make_entry(3, elo=1200, games=200, lineage=None)
        promoter._topk_streaks[3] = 0
        assert promoter.should_promote(no_lin, frontier, 100) is True
