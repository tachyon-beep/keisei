"""Tests for MatchScheduler -- role-weighted opponent selection."""

from collections import Counter

import pytest

from keisei.config import MatchSchedulerConfig, PriorityScorerConfig
from keisei.training.opponent_store import OpponentEntry, Role, EntryStatus
from keisei.training.match_scheduler import MatchScheduler
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
        counts = Counter()
        for _ in range(1000):
            entry = sched.sample_for_learner(full_entries)
            counts[entry.role] += 1
        assert 400 < counts[Role.DYNAMIC] < 600
        assert 200 < counts[Role.FRONTIER_STATIC] < 400
        assert 100 < counts[Role.RECENT_FIXED] < 300

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


class TestPriorityRound:
    def test_generate_round_returns_priority_sorted(self):
        scorer = PriorityScorer(PriorityScorerConfig())
        scheduler = _make_scheduler(priority_scorer=scorer)
        e1 = _make_entry(1, Role.DYNAMIC, elo=1000.0)
        e2 = _make_entry(2, Role.DYNAMIC, elo=1050.0)
        e3 = _make_entry(3, Role.DYNAMIC, elo=1500.0)
        entries = [e1, e2, e3]
        pairings = scheduler.generate_round(entries)
        scores = [scorer.score(a, b) for a, b in pairings]
        assert scores == sorted(scores, reverse=True)

    def test_generate_round_without_scorer_still_works(self):
        scheduler = _make_scheduler()
        entries = [_make_entry(i, Role.DYNAMIC) for i in range(1, 5)]
        pairings = scheduler.generate_round(entries)
        assert len(pairings) == 6
