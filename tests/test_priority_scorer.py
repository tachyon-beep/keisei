"""Tests for PriorityScorer -- matchup informativeness ranking."""

from __future__ import annotations

import pytest

from keisei.config import PriorityScorerConfig
from keisei.training.opponent_store import EntryStatus, OpponentEntry, Role
from keisei.training.priority_scorer import PriorityScorer


def _make_entry(
    id: int,
    role: Role = Role.DYNAMIC,
    elo: float = 1000.0,
    lineage: str | None = None,
    parent_id: int | None = None,
) -> OpponentEntry:
    return OpponentEntry(
        id=id,
        display_name=f"e{id}",
        architecture="resnet",
        model_params={},
        checkpoint_path=f"/tmp/{id}.pt",
        elo_rating=elo,
        created_epoch=0,
        games_played=10,
        created_at="2026-01-01",
        flavour_facts=[],
        role=role,
        status=EntryStatus.ACTIVE,
        lineage_group=lineage,
        parent_entry_id=parent_id,
    )


class TestScore:
    def test_under_sampled_pair_scores_higher(self):
        scorer = PriorityScorer(PriorityScorerConfig())
        a, b, c = _make_entry(1), _make_entry(2), _make_entry(3)
        scorer.record_result(a.id, b.id)
        scorer.record_result(a.id, b.id)
        scorer.record_result(a.id, b.id)
        score_ab = scorer.score(a, b)
        score_ac = scorer.score(a, c)
        assert score_ac > score_ab

    def test_uncertainty_bonus_for_close_elo(self):
        scorer = PriorityScorer(PriorityScorerConfig())
        a = _make_entry(1, elo=1000.0)
        b_close = _make_entry(2, elo=1050.0)
        b_far = _make_entry(3, elo=1200.0)
        score_close = scorer.score(a, b_close)
        score_far = scorer.score(a, b_far)
        assert score_close > score_far

    def test_recent_fixed_bonus(self):
        scorer = PriorityScorer(PriorityScorerConfig())
        a = _make_entry(1, role=Role.DYNAMIC)
        b_rf = _make_entry(2, role=Role.RECENT_FIXED)
        b_dyn = _make_entry(3, role=Role.DYNAMIC)
        score_rf = scorer.score(a, b_rf)
        score_dyn = scorer.score(a, b_dyn)
        assert score_rf > score_dyn

    def test_repeat_penalty(self):
        scorer = PriorityScorer(PriorityScorerConfig())
        a, b = _make_entry(1), _make_entry(2)
        score_before = scorer.score(a, b)
        scorer.record_round_result(a.id, b.id)
        scorer.advance_round()
        score_after = scorer.score(a, b)
        assert score_after < score_before

    def test_lineage_penalty_parent_child(self):
        scorer = PriorityScorer(PriorityScorerConfig())
        parent = _make_entry(1, lineage="lin-1")
        child = _make_entry(2, lineage="lin-1", parent_id=1)
        unrelated = _make_entry(3, lineage="lin-2")
        score_related = scorer.score(parent, child)
        score_unrelated = scorer.score(parent, unrelated)
        assert score_unrelated > score_related

    def test_lineage_penalty_same_group(self):
        scorer = PriorityScorer(PriorityScorerConfig())
        a = _make_entry(1, lineage="lin-1")
        sibling = _make_entry(2, lineage="lin-1")
        unrelated = _make_entry(3, lineage="lin-2")
        score_sibling = scorer.score(a, sibling)
        score_unrelated = scorer.score(a, unrelated)
        assert score_unrelated > score_sibling

    def test_diversity_bonus_cross_lineage(self):
        scorer = PriorityScorer(PriorityScorerConfig())
        a = _make_entry(1, lineage="lin-1")
        same = _make_entry(2, lineage="lin-1")
        diff = _make_entry(3, lineage="lin-2")
        score_same = scorer.score(a, same)
        score_diff = scorer.score(a, diff)
        assert score_diff > score_same

    def test_all_weights_zero_produces_zero(self):
        cfg = PriorityScorerConfig(
            under_sample_weight=0.0,
            uncertainty_weight=0.0,
            recent_fixed_bonus=0.0,
            diversity_weight=0.0,
            match_class_weight=0.0,
            repeat_penalty=0.0,
            lineage_penalty=0.0,
        )
        scorer = PriorityScorer(cfg)
        a, b = _make_entry(1), _make_entry(2)
        assert scorer.score(a, b) == 0.0

    def test_repeat_window_slides(self):
        cfg = PriorityScorerConfig(repeat_window_rounds=2)
        scorer = PriorityScorer(cfg)
        a, b = _make_entry(1), _make_entry(2)

        # Round 1: record (a,b), advance → history = [{(1,2)}]
        scorer.record_round_result(a.id, b.id)
        scorer.advance_round()
        # Round 2: record (a,b), advance → history = [{(1,2)}, {(1,2)}]
        scorer.record_round_result(a.id, b.id)
        scorer.advance_round()
        two_repeat_score = scorer.score(a, b)

        # Round 3: empty round, advance → history = [{(1,2)}, {}]
        # Round 1's entry was evicted by deque maxlen=2
        scorer.advance_round()
        one_repeat_score = scorer.score(a, b)

        # After eviction, penalty should be lighter (1 repeat vs 2)
        assert one_repeat_score > two_repeat_score, (
            "After window slide, repeat count should decrease from 2 to 1"
        )

        # Compared to a fresh scorer, there should still be SOME penalty
        fresh_score = PriorityScorer(cfg).score(a, b)
        assert fresh_score > one_repeat_score


    def test_uncertainty_bonus_at_100_boundary(self):
        """Two entries exactly 100 Elo apart -> bonus is 0.0 (threshold is < 100, not <= 100)."""
        scorer = PriorityScorer(PriorityScorerConfig())
        a = _make_entry(1, elo=1000.0)
        b_at_100 = _make_entry(2, elo=1100.0)  # exactly 100 apart
        b_at_99 = _make_entry(3, elo=1099.0)  # 99 apart

        # Exactly 100 apart: abs(1000-1100) = 100, 100 < 100 is False -> bonus 0.0
        assert scorer._uncertainty_bonus(a, b_at_100) == 0.0
        # 99 apart: abs(1000-1099) = 99, 99 < 100 is True -> bonus 1.0
        assert scorer._uncertainty_bonus(a, b_at_99) == 1.0

    def test_lineage_diversity_none_lineage(self):
        """Entry with lineage_group=None paired with any -> returns 1.0 (optimistic default)."""
        scorer = PriorityScorer(PriorityScorerConfig())
        none_lin = _make_entry(1, lineage=None)
        with_lin = _make_entry(2, lineage="lin-1")
        both_none = _make_entry(3, lineage=None)

        assert scorer._lineage_diversity(none_lin, with_lin) == 1.0
        assert scorer._lineage_diversity(with_lin, none_lin) == 1.0
        assert scorer._lineage_diversity(none_lin, both_none) == 1.0

    def test_lineage_closeness_b_is_parent(self):
        """b.parent_entry_id == a.id returns same as a.parent_entry_id == b.id (1.0)."""
        scorer = PriorityScorer(PriorityScorerConfig())
        parent = _make_entry(1, lineage="lin-1")
        child = _make_entry(2, lineage="lin-1", parent_id=1)

        # a is parent, b is child (b.parent_entry_id == a.id)
        assert scorer._lineage_closeness(parent, child) == 1.0
        # a is child, b is parent (a.parent_entry_id == b.id)
        assert scorer._lineage_closeness(child, parent) == 1.0


class TestScoreRound:
    def test_returns_sorted_by_priority_descending(self):
        scorer = PriorityScorer(PriorityScorerConfig())
        a = _make_entry(1, elo=1000.0)
        b = _make_entry(2, elo=1050.0)
        c = _make_entry(3, elo=1500.0)
        pairings = [(a, b), (a, c), (b, c)]
        sorted_pairings = scorer.sort_by_priority(pairings)
        scores = [scorer.score(p[0], p[1]) for p in sorted_pairings]
        assert scores == sorted(scores, reverse=True)

    def test_sort_by_priority_stable(self):
        """Two pairings with identical scores -> order is deterministic."""
        # Use all-zero weights so every pairing scores 0.0
        cfg = PriorityScorerConfig(
            under_sample_weight=0.0,
            uncertainty_weight=0.0,
            recent_fixed_bonus=0.0,
            diversity_weight=0.0,
            repeat_penalty=0.0,
            lineage_penalty=0.0,
        )
        scorer = PriorityScorer(cfg)
        a = _make_entry(1)
        b = _make_entry(2)
        c = _make_entry(3)
        pairings = [(a, b), (a, c), (b, c)]

        result1 = scorer.sort_by_priority(pairings)
        result2 = scorer.sort_by_priority(pairings)
        # Same input, same scorer state -> identical output order
        assert [(x.id, y.id) for x, y in result1] == [(x.id, y.id) for x, y in result2]
