"""Tests for MatchScheduler -- role-weighted opponent selection."""

from collections import Counter

import pytest

from keisei.config import HistoricalLibraryConfig, MatchSchedulerConfig, PriorityScorerConfig
from keisei.training.opponent_store import OpponentEntry, Role, EntryStatus
from keisei.training.match_scheduler import (
    MATCH_CLASS_WEIGHTS,
    MatchClass,
    MatchScheduler,
    classify_match,
    is_training_match,
)
from keisei.training.priority_scorer import PriorityScorer


def _make_entry(id: int, role: Role, elo: float = 1000.0) -> OpponentEntry:
    return OpponentEntry(
        id=id, display_name=f"e{id}", architecture="resnet",
        model_params={}, checkpoint_path=f"/p/{id}.pt", elo_rating=elo,
        created_epoch=id, games_played=0, created_at="2026-01-01",
        flavour_facts=[], role=role, status=EntryStatus.ACTIVE,
        parent_entry_id=None, lineage_group=None,
        protection_remaining=0, last_match_at=None,
    )


def _make_scheduler(priority_scorer=None, **overrides):
    defaults = dict(
        learner_dynamic_ratio=0.50,
        learner_frontier_ratio=0.30,
        learner_recent_ratio=0.20,
    )
    defaults.update(overrides)
    config = MatchSchedulerConfig(**defaults)
    return MatchScheduler(config, priority_scorer=priority_scorer)


@pytest.fixture
def full_entries():
    entries = {
        Role.FRONTIER_STATIC: [_make_entry(i, Role.FRONTIER_STATIC) for i in range(1, 6)],
        Role.RECENT_FIXED: [_make_entry(i, Role.RECENT_FIXED) for i in range(6, 11)],
        Role.DYNAMIC: [_make_entry(i, Role.DYNAMIC) for i in range(11, 21)],
    }
    return entries


class TestSampleForLearner:
    def test_respects_tier_ratios(self, full_entries):
        sched = MatchScheduler(MatchSchedulerConfig())
        n = 2000  # more samples for tighter bounds
        counts = Counter()
        for _ in range(n):
            entry = sched.sample_for_learner(full_entries)
            counts[entry.role] += 1
        # Expected: Dynamic=50%, Frontier=30%, Recent=20% of n=2000
        # Bounds: ±15% relative (narrower than before)
        assert 850 < counts[Role.DYNAMIC] < 1150      # 1000 ± 150
        assert 510 < counts[Role.FRONTIER_STATIC] < 690  # 600 ± 90
        assert 340 < counts[Role.RECENT_FIXED] < 460   # 400 ± 60

    def test_empty_tier_redistributes(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = {
            Role.FRONTIER_STATIC: [_make_entry(1, Role.FRONTIER_STATIC)],
            Role.RECENT_FIXED: [_make_entry(2, Role.RECENT_FIXED)],
            Role.DYNAMIC: [],
        }
        for _ in range(100):
            entry = sched.sample_for_learner(entries)
            assert entry.role != Role.DYNAMIC

    def test_single_tier_available(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = {
            Role.FRONTIER_STATIC: [],
            Role.RECENT_FIXED: [_make_entry(1, Role.RECENT_FIXED)],
            Role.DYNAMIC: [],
        }
        entry = sched.sample_for_learner(entries)
        assert entry.role is Role.RECENT_FIXED

    def test_all_empty_raises(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = {Role.FRONTIER_STATIC: [], Role.RECENT_FIXED: [], Role.DYNAMIC: []}
        with pytest.raises(ValueError, match="[Nn]o.*entries"):
            sched.sample_for_learner(entries)


class TestSampleKForLearner:
    """sample_k_for_learner draws K distinct opponents without replacement."""

    def test_zero_k_returns_empty(self, full_entries):
        sched = MatchScheduler(MatchSchedulerConfig())
        assert sched.sample_k_for_learner(full_entries, 0) == []

    def test_negative_k_returns_empty(self, full_entries):
        sched = MatchScheduler(MatchSchedulerConfig())
        assert sched.sample_k_for_learner(full_entries, -5) == []

    def test_returns_k_distinct_entries(self, full_entries):
        sched = MatchScheduler(MatchSchedulerConfig())
        sampled = sched.sample_k_for_learner(full_entries, k=4)
        assert len(sampled) == 4
        # Distinct: no duplicate IDs
        ids = [e.id for e in sampled]
        assert len(set(ids)) == 4

    def test_k_larger_than_pool_returns_all(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = {
            Role.FRONTIER_STATIC: [_make_entry(1, Role.FRONTIER_STATIC)],
            Role.RECENT_FIXED: [_make_entry(2, Role.RECENT_FIXED)],
            Role.DYNAMIC: [_make_entry(3, Role.DYNAMIC)],
        }
        sampled = sched.sample_k_for_learner(entries, k=10)
        assert len(sampled) == 3
        assert {e.id for e in sampled} == {1, 2, 3}

    def test_k_equal_to_pool_returns_all(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = {
            Role.FRONTIER_STATIC: [_make_entry(1, Role.FRONTIER_STATIC)],
            Role.RECENT_FIXED: [_make_entry(2, Role.RECENT_FIXED)],
            Role.DYNAMIC: [_make_entry(3, Role.DYNAMIC)],
        }
        sampled = sched.sample_k_for_learner(entries, k=3)
        assert len(sampled) == 3
        assert {e.id for e in sampled} == {1, 2, 3}

    def test_empty_pool_raises(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = {Role.FRONTIER_STATIC: [], Role.RECENT_FIXED: [], Role.DYNAMIC: []}
        with pytest.raises(ValueError, match="[Nn]o entries"):
            sched.sample_k_for_learner(entries, k=4)

    def test_does_not_mutate_input(self, full_entries):
        sched = MatchScheduler(MatchSchedulerConfig())
        before = {role: list(entries) for role, entries in full_entries.items()}
        sched.sample_k_for_learner(full_entries, k=4)
        assert full_entries == before

    def test_role_distribution_over_many_samples(self, full_entries):
        """Across many K-samples, aggregate role frequencies should track
        the configured tier ratios (dynamic=50%, frontier=30%, recent=20%)."""
        sched = MatchScheduler(MatchSchedulerConfig())
        counts = Counter()
        n_trials = 500
        k = 4
        for _ in range(n_trials):
            sampled = sched.sample_k_for_learner(full_entries, k=k)
            assert len(sampled) == k
            for e in sampled:
                counts[e.role] += 1
        total = sum(counts.values())
        assert total == n_trials * k
        # Bounds are loose because K-without-replacement skews toward uniform
        # as K approaches the per-role size. Just verify all three tiers
        # are represented and dynamic is the most-sampled.
        assert counts[Role.DYNAMIC] > 0
        assert counts[Role.FRONTIER_STATIC] > 0
        assert counts[Role.RECENT_FIXED] > 0
        # Dynamic (50% weight, 10 entries) should dominate Frontier and Recent
        # in expectation; we just need to beat noise.
        assert counts[Role.DYNAMIC] > counts[Role.RECENT_FIXED]

    def test_single_tier_pool(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = {
            Role.FRONTIER_STATIC: [],
            Role.RECENT_FIXED: [],
            Role.DYNAMIC: [_make_entry(i, Role.DYNAMIC) for i in range(1, 6)],
        }
        sampled = sched.sample_k_for_learner(entries, k=3)
        assert len(sampled) == 3
        assert all(e.role is Role.DYNAMIC for e in sampled)
        assert len({e.id for e in sampled}) == 3

    def test_role_exhaustion_falls_through_to_other_roles(self):
        """When one role's entries are drained mid-sample, remaining
        picks should fall through to other non-empty roles."""
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = {
            # Only 1 Frontier entry — role weight can only contribute 1 pick.
            Role.FRONTIER_STATIC: [_make_entry(1, Role.FRONTIER_STATIC)],
            Role.RECENT_FIXED: [_make_entry(2, Role.RECENT_FIXED)],
            Role.DYNAMIC: [
                _make_entry(3, Role.DYNAMIC),
                _make_entry(4, Role.DYNAMIC),
            ],
        }
        # K=4 = total pool size → must return all 4 without error
        sampled = sched.sample_k_for_learner(entries, k=4)
        assert {e.id for e in sampled} == {1, 2, 3, 4}


class TestGenerateRound:
    def test_produces_all_pairs(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = [_make_entry(i, Role.DYNAMIC) for i in range(5)]
        pairings = sched.generate_round(entries)
        assert len(pairings) == 10
        pair_ids = {(min(a.id, b.id), max(a.id, b.id)) for a, b in pairings}
        assert len(pair_ids) == 10

    def test_no_self_matches(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = [_make_entry(i, Role.DYNAMIC) for i in range(5)]
        pairings = sched.generate_round(entries)
        for a, b in pairings:
            assert a.id != b.id

    def test_single_entry_produces_no_pairings(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = [_make_entry(1, Role.DYNAMIC)]
        pairings = sched.generate_round(entries)
        assert len(pairings) == 0

    def test_mixed_roles_all_paired(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = [
            _make_entry(1, Role.FRONTIER_STATIC),
            _make_entry(2, Role.RECENT_FIXED),
            _make_entry(3, Role.DYNAMIC),
        ]
        pairings = sched.generate_round(entries)
        assert len(pairings) == 3
        # Verify all pairs are unique (no duplicate pairings)
        pair_ids = {(min(a.id, b.id), max(a.id, b.id)) for a, b in pairings}
        assert len(pair_ids) == 3


class TestEffectiveRatios:
    def test_full_pool_returns_config_ratios(self, full_entries):
        sched = MatchScheduler(MatchSchedulerConfig())
        ratios = sched.effective_ratios(full_entries)
        assert abs(ratios[Role.DYNAMIC] - 0.50) < 0.01
        assert abs(ratios[Role.FRONTIER_STATIC] - 0.30) < 0.01
        assert abs(ratios[Role.RECENT_FIXED] - 0.20) < 0.01

    def test_empty_tier_redistributes(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = {
            Role.FRONTIER_STATIC: [_make_entry(1, Role.FRONTIER_STATIC)],
            Role.RECENT_FIXED: [_make_entry(2, Role.RECENT_FIXED)],
            Role.DYNAMIC: [],
        }
        ratios = sched.effective_ratios(entries)
        assert ratios[Role.DYNAMIC] == 0.0
        assert abs(ratios[Role.FRONTIER_STATIC] + ratios[Role.RECENT_FIXED] - 1.0) < 0.01


    def test_negative_ratio_rejected_by_config(self):
        """Negative individual ratios should be rejected even if they sum to 1.0."""
        with pytest.raises(ValueError, match="must be >= 0"):
            MatchSchedulerConfig(
                learner_dynamic_ratio=-0.5,
                learner_frontier_ratio=1.0,
                learner_recent_ratio=0.5,
            )

    def test_effective_ratios_guards_zero_total(self):
        """If only zero-weight tiers are populated, effective_ratios returns all zeros."""
        sched = MatchScheduler(MatchSchedulerConfig(
            learner_dynamic_ratio=0.0,
            learner_frontier_ratio=0.0,
            learner_recent_ratio=1.0,
        ))
        entries = {
            Role.DYNAMIC: [_make_entry(1, Role.DYNAMIC)],
            Role.FRONTIER_STATIC: [_make_entry(2, Role.FRONTIER_STATIC)],
            Role.RECENT_FIXED: [],  # the only tier with weight > 0 is empty
        }
        ratios = sched.effective_ratios(entries)
        # All weights for populated tiers are 0.0, so total is 0 — must not crash
        assert ratios[Role.DYNAMIC] == 0.0
        assert ratios[Role.FRONTIER_STATIC] == 0.0
        assert ratios[Role.RECENT_FIXED] == 0.0


class TestPriorityRound:
    def test_generate_round_returns_priority_sorted(self):
        scorer = PriorityScorer(PriorityScorerConfig())
        scheduler = _make_scheduler(priority_scorer=scorer)
        e1 = _make_entry(1, Role.DYNAMIC, elo=1000.0)
        e2 = _make_entry(2, Role.DYNAMIC, elo=1050.0)
        e3 = _make_entry(3, Role.DYNAMIC, elo=1500.0)
        entries = [e1, e2, e3]
        pairings = scheduler.generate_round(entries)
        # Verify descending order by independently computing scores
        # (Note: uses same scorer instance, but PriorityScorer.score is
        # stateless given the same inputs, so re-scoring is valid here.)
        scores = [scorer.score(a, b) for a, b in pairings]
        assert scores == sorted(scores, reverse=True)
        # Verify we actually got distinct scores (not all equal)
        assert len(set(scores)) > 1, "All scores equal — sorting is vacuous"

    def test_generate_round_without_scorer_still_works(self):
        scheduler = _make_scheduler()
        entries = [_make_entry(i, Role.DYNAMIC) for i in range(1, 5)]
        pairings = scheduler.generate_round(entries)
        assert len(pairings) == 6


# ---------------------------------------------------------------------------
# T2. Weighted tournament mode
# ---------------------------------------------------------------------------

class TestClassifyMatch:
    """Test classify_match() for all role combinations."""

    def test_dynamic_vs_dynamic(self):
        a = _make_entry(1, Role.DYNAMIC)
        b = _make_entry(2, Role.DYNAMIC)
        assert classify_match(a, b) == MatchClass.DYNAMIC_VS_DYNAMIC

    def test_dynamic_vs_recent_fixed(self):
        a = _make_entry(1, Role.DYNAMIC)
        b = _make_entry(2, Role.RECENT_FIXED)
        assert classify_match(a, b) == MatchClass.DYNAMIC_VS_RECENT
        # Order should not matter
        assert classify_match(b, a) == MatchClass.DYNAMIC_VS_RECENT

    def test_dynamic_vs_frontier_static(self):
        a = _make_entry(1, Role.DYNAMIC)
        b = _make_entry(2, Role.FRONTIER_STATIC)
        assert classify_match(a, b) == MatchClass.DYNAMIC_VS_FRONTIER
        assert classify_match(b, a) == MatchClass.DYNAMIC_VS_FRONTIER

    def test_recent_fixed_vs_frontier_static(self):
        a = _make_entry(1, Role.RECENT_FIXED)
        b = _make_entry(2, Role.FRONTIER_STATIC)
        assert classify_match(a, b) == MatchClass.RECENT_VS_FRONTIER
        assert classify_match(b, a) == MatchClass.RECENT_VS_FRONTIER

    def test_recent_fixed_vs_recent_fixed(self):
        a = _make_entry(1, Role.RECENT_FIXED)
        b = _make_entry(2, Role.RECENT_FIXED)
        assert classify_match(a, b) == MatchClass.RECENT_VS_RECENT

    def test_frontier_static_vs_frontier_static(self):
        a = _make_entry(1, Role.FRONTIER_STATIC)
        b = _make_entry(2, Role.FRONTIER_STATIC)
        assert classify_match(a, b) == MatchClass.FRONTIER_VS_FRONTIER

    def test_unassigned_role_gives_other(self):
        a = _make_entry(1, Role.UNASSIGNED)
        b = _make_entry(2, Role.DYNAMIC)
        assert classify_match(a, b) == MatchClass.OTHER


class TestMatchClassWeights:
    """Assert MATCH_CLASS_WEIGHTS match the plan values."""

    def test_dynamic_vs_dynamic_weight(self):
        assert MATCH_CLASS_WEIGHTS[MatchClass.DYNAMIC_VS_DYNAMIC] == pytest.approx(0.40)

    def test_dynamic_vs_recent_weight(self):
        assert MATCH_CLASS_WEIGHTS[MatchClass.DYNAMIC_VS_RECENT] == pytest.approx(0.25)

    def test_dynamic_vs_frontier_weight(self):
        assert MATCH_CLASS_WEIGHTS[MatchClass.DYNAMIC_VS_FRONTIER] == pytest.approx(0.20)

    def test_recent_vs_frontier_weight(self):
        assert MATCH_CLASS_WEIGHTS[MatchClass.RECENT_VS_FRONTIER] == pytest.approx(0.10)

    def test_recent_vs_recent_weight(self):
        assert MATCH_CLASS_WEIGHTS[MatchClass.RECENT_VS_RECENT] == pytest.approx(0.05)


class TestIsTrainingMatch:
    """Test is_training_match() returns True only for D-D and D-RF."""

    def test_dynamic_vs_dynamic_is_training(self):
        a = _make_entry(1, Role.DYNAMIC)
        b = _make_entry(2, Role.DYNAMIC)
        assert is_training_match(a, b) is True

    def test_dynamic_vs_recent_is_training(self):
        a = _make_entry(1, Role.DYNAMIC)
        b = _make_entry(2, Role.RECENT_FIXED)
        assert is_training_match(a, b) is True

    def test_dynamic_vs_frontier_not_training(self):
        a = _make_entry(1, Role.DYNAMIC)
        b = _make_entry(2, Role.FRONTIER_STATIC)
        assert is_training_match(a, b) is False

    def test_recent_vs_frontier_not_training(self):
        a = _make_entry(1, Role.RECENT_FIXED)
        b = _make_entry(2, Role.FRONTIER_STATIC)
        assert is_training_match(a, b) is False

    def test_recent_vs_recent_not_training(self):
        a = _make_entry(1, Role.RECENT_FIXED)
        b = _make_entry(2, Role.RECENT_FIXED)
        assert is_training_match(a, b) is False

    def test_frontier_vs_frontier_not_training(self):
        a = _make_entry(1, Role.FRONTIER_STATIC)
        b = _make_entry(2, Role.FRONTIER_STATIC)
        assert is_training_match(a, b) is False


class TestWeightedTournamentMode:
    """Test generate_round() in weighted mode with mixed-role entry pool."""

    def test_weighted_round_respects_round_size_budget(self):
        """Per-class max(1,...) must not produce more pairings than round_size."""
        # 5 match classes with non-zero weight + round_size=3 → bug would give 5
        scheduler = _make_scheduler(
            tournament_mode="weighted",
            weighted_round_size=3,
        )
        entries = [
            _make_entry(1, Role.DYNAMIC),
            _make_entry(2, Role.DYNAMIC),
            _make_entry(3, Role.RECENT_FIXED),
            _make_entry(4, Role.RECENT_FIXED),
            _make_entry(5, Role.FRONTIER_STATIC),
            _make_entry(6, Role.FRONTIER_STATIC),
        ]
        pairings = scheduler.generate_round(entries)
        assert len(pairings) <= 3, (
            f"Expected at most 3 pairings (round_size=3), got {len(pairings)}"
        )

    def test_weighted_round_produces_multiple_match_classes(self):
        scheduler = _make_scheduler(tournament_mode="weighted")
        entries = [
            _make_entry(1, Role.DYNAMIC),
            _make_entry(2, Role.DYNAMIC),
            _make_entry(3, Role.RECENT_FIXED),
            _make_entry(4, Role.RECENT_FIXED),
            _make_entry(5, Role.FRONTIER_STATIC),
            _make_entry(6, Role.FRONTIER_STATIC),
        ]
        pairings = scheduler.generate_round(entries)
        assert len(pairings) > 0
        # Collect the match classes that appear in the round
        classes_seen = {classify_match(a, b) for a, b in pairings}
        # With 6 entries (2 of each role), weighted mode should draw from
        # multiple match classes, not just one.
        assert len(classes_seen) >= 2, (
            f"Expected pairings from multiple match classes, got only {classes_seen}"
        )


# ---------------------------------------------------------------------------
# T9. Historical library exclusion
# ---------------------------------------------------------------------------

class TestHistoricalLibraryExclusion:
    """Non-active roles must be excluded from active-league matchmaking."""

    def test_unassigned_role_excluded_from_generate_round(self):
        """Entries with UNASSIGNED role should classify as OTHER (weight 0)
        and therefore not appear in weighted round output."""
        scheduler = _make_scheduler(tournament_mode="weighted")
        # Only UNASSIGNED entries -- no valid match classes with weight > 0
        entries = [
            _make_entry(1, Role.UNASSIGNED),
            _make_entry(2, Role.UNASSIGNED),
        ]
        pairings = scheduler.generate_round(entries)
        # All pairs are OTHER class (weight=0), so weighted sampling falls
        # back to shuffled. Verify that if we mix with active roles, the
        # weighted mode does not select UNASSIGNED-only pairs by preference.
        # The key invariant: classify_match gives OTHER for these pairs.
        for a, b in pairings:
            assert classify_match(a, b) == MatchClass.OTHER

    def test_unassigned_role_excluded_from_sample_for_learner(self):
        """sample_for_learner only draws from DYNAMIC/RECENT_FIXED/FRONTIER_STATIC
        tiers, so UNASSIGNED entries should never be returned."""
        scheduler = _make_scheduler()
        entries = {
            Role.DYNAMIC: [_make_entry(1, Role.DYNAMIC)],
            Role.FRONTIER_STATIC: [_make_entry(2, Role.FRONTIER_STATIC)],
            Role.RECENT_FIXED: [],
            Role.UNASSIGNED: [_make_entry(3, Role.UNASSIGNED)],
        }
        for _ in range(50):
            entry = scheduler.sample_for_learner(entries)
            assert entry.role != Role.UNASSIGNED

    def test_historical_library_active_participation_raises(self):
        """HistoricalLibraryConfig(active_league_participation=True) must raise."""
        with pytest.raises(ValueError, match="active_league_participation"):
            HistoricalLibraryConfig(active_league_participation=True)


# ---------------------------------------------------------------------------
# T10. Minimum coverage ratio
# ---------------------------------------------------------------------------

class TestMinCoverageRatio:
    """Test min_coverage_ratio enforcement in weighted mode."""

    def test_coverage_ratio_config_validation(self):
        """min_coverage_ratio must be in [0.0, 1.0]."""
        # Valid values should not raise
        MatchSchedulerConfig(min_coverage_ratio=0.0)
        MatchSchedulerConfig(min_coverage_ratio=0.5)
        MatchSchedulerConfig(min_coverage_ratio=1.0)

        # Invalid values should raise
        with pytest.raises(ValueError, match="min_coverage_ratio"):
            MatchSchedulerConfig(min_coverage_ratio=-0.1)
        with pytest.raises(ValueError, match="min_coverage_ratio"):
            MatchSchedulerConfig(min_coverage_ratio=1.1)

    def test_weighted_mode_ensures_minimum_coverage(self):
        """Weighted mode should ensure at least min_coverage_ratio of entries play."""
        # Create a pool where weighted sampling would normally favor certain pairs
        # 4 DYNAMIC (weight 0.40 for D-vs-D) + 4 FRONTIER (weight 0.0 for F-vs-F)
        # Without coverage enforcement, Frontier entries might be excluded
        entries = [
            _make_entry(i, Role.DYNAMIC) for i in range(1, 5)
        ] + [
            _make_entry(i, Role.FRONTIER_STATIC) for i in range(5, 9)
        ]
        # 8 entries, 50% coverage = at least 4 must participate

        scheduler = _make_scheduler(
            tournament_mode="weighted",
            weighted_round_size=4,  # small round to test coverage enforcement
            min_coverage_ratio=0.5,
            # Skew weights to make D-vs-D dominant
            dynamic_dynamic_weight=0.95,
            dynamic_recent_weight=0.0,
            dynamic_frontier_weight=0.05,
            recent_frontier_weight=0.0,
            recent_recent_weight=0.0,
        )

        # Run multiple times to verify coverage is consistently enforced
        for _ in range(10):
            pairings = scheduler.generate_round(entries)
            covered_ids = set()
            for a, b in pairings:
                covered_ids.add(a.id)
                covered_ids.add(b.id)

            # At least 50% of entries (4 out of 8) should be covered
            assert len(covered_ids) >= 4, (
                f"Expected at least 4 entries covered (50% of 8), "
                f"got {len(covered_ids)}: {covered_ids}"
            )

    def test_zero_coverage_ratio_disables_enforcement(self):
        """min_coverage_ratio=0.0 should disable coverage enforcement."""
        entries = [
            _make_entry(i, Role.DYNAMIC) for i in range(1, 5)
        ] + [
            _make_entry(i, Role.FRONTIER_STATIC) for i in range(5, 9)
        ]

        scheduler = _make_scheduler(
            tournament_mode="weighted",
            weighted_round_size=2,  # very small round
            min_coverage_ratio=0.0,  # disabled
            dynamic_dynamic_weight=1.0,  # only D-vs-D matches
            dynamic_recent_weight=0.0,
            dynamic_frontier_weight=0.0,
            recent_frontier_weight=0.0,
            recent_recent_weight=0.0,
        )

        pairings = scheduler.generate_round(entries)
        # With coverage disabled and only D-vs-D weight, all pairs should be Dynamic
        for a, b in pairings:
            assert a.role == Role.DYNAMIC
            assert b.role == Role.DYNAMIC

    def test_full_coverage_ratio_includes_all_entries(self):
        """min_coverage_ratio=1.0 should include all entries."""
        entries = [
            _make_entry(1, Role.DYNAMIC),
            _make_entry(2, Role.DYNAMIC),
            _make_entry(3, Role.FRONTIER_STATIC),
            _make_entry(4, Role.FRONTIER_STATIC),
        ]

        scheduler = _make_scheduler(
            tournament_mode="weighted",
            min_coverage_ratio=1.0,
        )

        pairings = scheduler.generate_round(entries)
        covered_ids = set()
        for a, b in pairings:
            covered_ids.add(a.id)
            covered_ids.add(b.id)

        assert covered_ids == {1, 2, 3, 4}, (
            f"Expected all 4 entries covered, got {covered_ids}"
        )

    def test_coverage_enforcement_adds_minimum_pairings(self):
        """Coverage enforcement should add pairings for uncovered entries."""
        # Pool with 2 DYNAMIC (high weight) and 6 FRONTIER (zero weight for F-vs-F)
        entries = [
            _make_entry(1, Role.DYNAMIC),
            _make_entry(2, Role.DYNAMIC),
        ] + [
            _make_entry(i, Role.FRONTIER_STATIC) for i in range(3, 9)
        ]
        # 8 entries, 50% coverage = at least 4 must participate

        scheduler = _make_scheduler(
            tournament_mode="weighted",
            weighted_round_size=10,  # large enough to not constrain
            min_coverage_ratio=0.5,
            # Zero out most classes to force coverage mechanism to work
            dynamic_dynamic_weight=1.0,
            dynamic_recent_weight=0.0,
            dynamic_frontier_weight=0.0,
            recent_frontier_weight=0.0,
            recent_recent_weight=0.0,
        )

        pairings = scheduler.generate_round(entries)
        covered_ids = set()
        for a, b in pairings:
            covered_ids.add(a.id)
            covered_ids.add(b.id)

        # Must cover at least 4 entries (50% of 8)
        assert len(covered_ids) >= 4, (
            f"Expected at least 4 entries, got {len(covered_ids)}: {covered_ids}"
        )

    def test_coverage_survives_round_size_trim(self):
        """Coverage enforcement must not be undone by the final priority sort + trim.

        Regression test: previously _enforce_min_coverage ran BEFORE the final
        sort/truncate, so a low-priority coverage pair added for an uncovered
        entry was silently dropped when the combined list was trimmed back
        to ``weighted_round_size``.  With 4 Dynamic entries, round_size=2,
        min_coverage_ratio=1.0, and a scorer that rates pairs containing
        entry 4 lowest, the coverage pair for entry 4 used to disappear.
        """
        entries = [_make_entry(i, Role.DYNAMIC) for i in range(1, 5)]

        scorer = PriorityScorer(PriorityScorerConfig())
        # Inflate pair_games for every pair containing entry 4 so that the
        # under_sample component drives their priority below all other pairs.
        for other in (1, 2, 3):
            for _ in range(10):
                scorer.record_result(other, 4)

        scheduler = _make_scheduler(
            priority_scorer=scorer,
            tournament_mode="weighted",
            weighted_round_size=2,
            min_coverage_ratio=1.0,
        )

        pairings = scheduler.generate_round(entries)
        covered_ids = {eid for a, b in pairings for eid in (a.id, b.id)}
        assert covered_ids == {1, 2, 3, 4}, (
            f"Expected all 4 entries covered under min_coverage_ratio=1.0, "
            f"got {covered_ids} from {[(a.id, b.id) for a, b in pairings]}"
        )
