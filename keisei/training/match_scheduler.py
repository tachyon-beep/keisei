"""MatchScheduler -- role-weighted opponent selection for the tiered pool."""

from __future__ import annotations

import random
from collections import deque
from enum import StrEnum
from typing import TYPE_CHECKING

from keisei.config import MatchSchedulerConfig
from keisei.training.opponent_store import OpponentEntry, Role

if TYPE_CHECKING:
    from keisei.training.priority_scorer import PriorityScorer


class TournamentMode(StrEnum):
    """How generate_round() builds the pairing list."""

    FULL = "full"          # All N*(N-1)/2 pairs, priority-sorted
    WEIGHTED = "weighted"  # Sample pairs according to §8.3 match-class ratios
    RANDOM = "random"      # All N*(N-1)/2 pairs, shuffled (no priority)


class MatchClass(StrEnum):
    """Match classification per §8.2 of tiered-opponent-pool design.

    Training classes produce gradient updates for Dynamic entries.
    Calibration classes produce Elo data only.
    """

    # Training matches
    DYNAMIC_VS_DYNAMIC = "dynamic_vs_dynamic"
    DYNAMIC_VS_RECENT = "dynamic_vs_recent"

    # Active-league calibration matches
    DYNAMIC_VS_FRONTIER = "dynamic_vs_frontier"
    RECENT_VS_FRONTIER = "recent_vs_frontier"
    RECENT_VS_RECENT = "recent_vs_recent"
    FRONTIER_VS_FRONTIER = "frontier_vs_frontier"

    # Catch-all for unexpected role combinations
    OTHER = "other"


# §8.3 recommended active-league mix ratios, used as priority weights.
MATCH_CLASS_WEIGHTS: dict[MatchClass, float] = {
    MatchClass.DYNAMIC_VS_DYNAMIC: 0.40,
    MatchClass.DYNAMIC_VS_RECENT: 0.25,
    MatchClass.DYNAMIC_VS_FRONTIER: 0.20,
    MatchClass.RECENT_VS_FRONTIER: 0.10,
    MatchClass.RECENT_VS_RECENT: 0.05,
    MatchClass.FRONTIER_VS_FRONTIER: 0.0,
    MatchClass.OTHER: 0.0,
}

TRAINING_CLASSES = frozenset({
    MatchClass.DYNAMIC_VS_DYNAMIC,
    MatchClass.DYNAMIC_VS_RECENT,
})


def classify_match(a: OpponentEntry, b: OpponentEntry) -> MatchClass:
    """Classify a pairing by role combination (§8.2)."""
    roles = frozenset({a.role, b.role})
    if roles == {Role.DYNAMIC}:
        return MatchClass.DYNAMIC_VS_DYNAMIC
    if roles == {Role.DYNAMIC, Role.RECENT_FIXED}:
        return MatchClass.DYNAMIC_VS_RECENT
    if roles == {Role.DYNAMIC, Role.FRONTIER_STATIC}:
        return MatchClass.DYNAMIC_VS_FRONTIER
    if roles == {Role.RECENT_FIXED, Role.FRONTIER_STATIC}:
        return MatchClass.RECENT_VS_FRONTIER
    if roles == {Role.RECENT_FIXED}:
        return MatchClass.RECENT_VS_RECENT
    if roles == {Role.FRONTIER_STATIC}:
        return MatchClass.FRONTIER_VS_FRONTIER
    return MatchClass.OTHER


def is_training_match(a: OpponentEntry, b: OpponentEntry) -> bool:
    """True if this pairing produces training data for Dynamic entries (§10.1)."""
    return classify_match(a, b) in TRAINING_CLASSES


def build_match_class_weights(config: MatchSchedulerConfig) -> dict[MatchClass, float]:
    """Build a match-class weights dict from config fields (§8.3)."""
    return {
        MatchClass.DYNAMIC_VS_DYNAMIC: config.dynamic_dynamic_weight,
        MatchClass.DYNAMIC_VS_RECENT: config.dynamic_recent_weight,
        MatchClass.DYNAMIC_VS_FRONTIER: config.dynamic_frontier_weight,
        MatchClass.RECENT_VS_FRONTIER: config.recent_frontier_weight,
        MatchClass.RECENT_VS_RECENT: config.recent_recent_weight,
        MatchClass.FRONTIER_VS_FRONTIER: 0.0,
        MatchClass.OTHER: 0.0,
    }


class MatchScheduler:
    def __init__(
        self,
        config: MatchSchedulerConfig,
        priority_scorer: PriorityScorer | None = None,
    ) -> None:
        self.config = config
        self._priority_scorer = priority_scorer
        self._match_class_weights = build_match_class_weights(config)
        # Rolling win-rate tracker per tier for challenge threshold (Fix 9).
        # Each deque stores recent (won: bool) outcomes for learner vs that tier.
        self._tier_outcomes: dict[Role, deque[bool]] = {
            role: deque(maxlen=config.challenge_window)
            for role in (Role.DYNAMIC, Role.FRONTIER_STATIC, Role.RECENT_FIXED)
        }

    @property
    def match_class_weights(self) -> dict[MatchClass, float]:
        """Config-driven match-class weights (§8.3)."""
        return self._match_class_weights

    @property
    def priority_scorer(self) -> PriorityScorer | None:
        """The PriorityScorer instance, or None if not configured."""
        return self._priority_scorer

    def record_learner_result(self, opponent_role: Role, won: bool) -> None:
        """Record a learner win/loss for challenge threshold tracking."""
        if opponent_role in self._tier_outcomes:
            self._tier_outcomes[opponent_role].append(won)

    def tier_win_rate(self, role: Role) -> float | None:
        """Rolling win rate against a tier, or None if insufficient data."""
        outcomes = self._tier_outcomes.get(role)
        if not outcomes or len(outcomes) < 10:
            return None
        return sum(outcomes) / len(outcomes)

    def sample_for_learner(
        self, entries_by_role: dict[Role, list[OpponentEntry]],
    ) -> OpponentEntry:
        ratios = self.effective_ratios(entries_by_role)
        non_empty = {r: w for r, w in ratios.items() if w > 0}
        if not non_empty:
            raise ValueError(
                "No entries available in any tier (entries may exist but none "
                "have roles matching configured tier ratios)"
            )
        roles = list(non_empty.keys())
        weights = [non_empty[r] for r in roles]
        chosen_role = random.choices(roles, weights=weights, k=1)[0]
        # Safe: effective_ratios() uses .get() to filter roles, so only roles
        # present in entries_by_role with non-empty lists survive into non_empty.
        return random.choice(entries_by_role[chosen_role])

    def sample_k_for_learner(
        self, entries_by_role: dict[Role, list[OpponentEntry]], k: int,
    ) -> list[OpponentEntry]:
        """Sample K distinct opponents via role-weighted sampling without replacement.

        Used at epoch start to pick the cohort of opponents the learner will
        face for one rollout epoch. Role weights come from effective_ratios()
        (same as sample_for_learner); within a role, entries are drawn
        uniformly at random. When a role's entries are exhausted mid-sample,
        its weight is redistributed to remaining non-empty roles.

        Returns fewer than K entries if the pool is smaller than K. Returns
        an empty list if k <= 0. Raises ValueError if the pool is empty.
        """
        if k <= 0:
            return []

        total_available = sum(len(v) for v in entries_by_role.values())
        if total_available == 0:
            raise ValueError(
                "No entries available in any tier — cannot sample opponents"
            )
        if k >= total_available:
            # Flatten all entries; order is implementation-defined but stable.
            return [e for entries in entries_by_role.values() for e in entries]

        # Shallow copy so we can drop sampled entries without mutating caller's dict.
        remaining: dict[Role, list[OpponentEntry]] = {
            role: list(entries) for role, entries in entries_by_role.items()
        }

        sampled: list[OpponentEntry] = []
        while len(sampled) < k:
            ratios = self.effective_ratios(remaining)
            non_empty = {
                r: w for r, w in ratios.items() if w > 0 and remaining.get(r)
            }
            if not non_empty:
                # Roles with positive weight are exhausted. Fall back to
                # uniform sampling across whatever is still in remaining —
                # this handles the edge case where effective_ratios() zeros
                # out all roles (e.g. learner dominates every tier) but we
                # still have entries the caller wants us to pick from.
                flat = [e for entries in remaining.values() for e in entries]
                if not flat:
                    break
                sampled.append(flat[random.randrange(len(flat))])
                # Remove sampled entry from remaining
                for entries in remaining.values():
                    if sampled[-1] in entries:
                        entries.remove(sampled[-1])
                        break
                continue
            roles = list(non_empty.keys())
            weights = [non_empty[r] for r in roles]
            chosen_role = random.choices(roles, weights=weights, k=1)[0]
            idx = random.randrange(len(remaining[chosen_role]))
            sampled.append(remaining[chosen_role].pop(idx))

        return sampled

    def generate_round(
        self, entries: list[OpponentEntry],
    ) -> list[tuple[OpponentEntry, OpponentEntry]]:
        """Generate pairings for a tournament round.

        Mode (from config.tournament_mode):
          - ``full``:     All N*(N-1)/2 pairs, priority-sorted if scorer available.
          - ``weighted``: Sample pairs according to §8.3 match-class ratios.
          - ``random``:   All N*(N-1)/2 pairs, shuffled (no priority scoring).
        """
        mode = TournamentMode(self.config.tournament_mode)

        if mode is TournamentMode.RANDOM:
            pairings = self._all_pairs(entries)
            random.shuffle(pairings)
            return pairings

        if mode is TournamentMode.FULL:
            pairings = self._all_pairs(entries)
            if self._priority_scorer is not None:
                return self._priority_scorer.sort_by_priority(pairings)
            random.shuffle(pairings)
            return pairings

        # WEIGHTED: sample according to §8.3 match-class ratios
        return self._weighted_sample(entries)

    @staticmethod
    def _all_pairs(
        entries: list[OpponentEntry],
    ) -> list[tuple[OpponentEntry, OpponentEntry]]:
        """Generate all N*(N-1)/2 unique pairings."""
        pairings: list[tuple[OpponentEntry, OpponentEntry]] = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                pairings.append((entries[i], entries[j]))
        return pairings

    def _weighted_sample(
        self, entries: list[OpponentEntry],
    ) -> list[tuple[OpponentEntry, OpponentEntry]]:
        """Sample pairings weighted by §8.3 match-class ratios.

        Groups all possible pairs by MatchClass, then draws from each class
        proportionally to MATCH_CLASS_WEIGHTS.  Within each class, pairs are
        priority-sorted (if scorer available) or shuffled.

        After weighted selection, enforces min_coverage_ratio by adding pairings
        for under-represented entries until the coverage threshold is met.
        """
        all_pairs = self._all_pairs(entries)
        if not all_pairs:
            return []

        # Group by match class
        buckets: dict[MatchClass, list[tuple[OpponentEntry, OpponentEntry]]] = {}
        for pair in all_pairs:
            mc = classify_match(pair[0], pair[1])
            buckets.setdefault(mc, []).append(pair)

        # Determine round size: 0 = auto (N pairings, one per entry on average)
        round_size = self.config.weighted_round_size
        if round_size <= 0:
            round_size = len(entries)

        # Distribute round_size across match classes proportionally
        weights = self._match_class_weights
        present_classes = {mc for mc in buckets if weights.get(mc, 0) > 0}
        if not present_classes:
            # No weighted classes available — fall back to shuffled full round
            random.shuffle(all_pairs)
            return all_pairs[:round_size]

        total_weight = sum(weights[mc] for mc in present_classes)

        result: list[tuple[OpponentEntry, OpponentEntry]] = []
        for mc in present_classes:
            pool = buckets.get(mc, [])
            if not pool:
                continue
            # Sort within class by priority (most informative first)
            if self._priority_scorer is not None:
                pool = self._priority_scorer.sort_by_priority(pool)
            else:
                random.shuffle(pool)
            # Allocate proportional share of round_size
            share = max(1, round(round_size * weights[mc] / total_weight))
            result.extend(pool[:share])

        # Final priority sort across all selected pairs, then enforce budget.
        # Coverage enforcement runs AFTER the trim so coverage-critical pairs
        # cannot be silently dropped when low-priority.
        if self._priority_scorer is not None:
            result = self._priority_scorer.sort_by_priority(result)
        else:
            random.shuffle(result)
        result = result[:round_size]

        # Enforce minimum coverage: ensure at least min_coverage_ratio of entries
        # appear in at least one pairing. This prevents models from being starved
        # of games when weighted sampling favors certain match classes. When the
        # budget is already saturated, evict the lowest-priority pair whose
        # removal does not reduce coverage.
        return self._enforce_min_coverage(entries, all_pairs, result, round_size)

    def _enforce_min_coverage(
        self,
        entries: list[OpponentEntry],
        all_pairs: list[tuple[OpponentEntry, OpponentEntry]],
        selected: list[tuple[OpponentEntry, OpponentEntry]],
        round_size: int,
    ) -> list[tuple[OpponentEntry, OpponentEntry]]:
        """Add pairings to ensure min_coverage_ratio of entries participate.

        Finds entries not covered by selected pairings and adds the best
        available pairing for each until the coverage threshold is met.

        When ``len(selected) >= round_size`` the round is already at budget;
        we make room by evicting the lowest-priority pair whose removal does
        not reduce coverage (i.e. both endpoints remain covered by another
        selected pair).  If no such evictable pair exists, we append anyway
        rather than silently violate the documented coverage invariant — a
        small overrun is preferable to undercovering.
        """
        if self.config.min_coverage_ratio <= 0.0:
            return selected

        all_entry_ids = {e.id for e in entries}
        min_covered = max(1, int(len(entries) * self.config.min_coverage_ratio + 0.5))

        covered_ids: set[int] = set()
        for a, b in selected:
            covered_ids.add(a.id)
            covered_ids.add(b.id)

        if len(covered_ids) >= min_covered:
            return selected

        def _canon(pair: tuple[OpponentEntry, OpponentEntry]) -> tuple[int, int]:
            return (min(pair[0].id, pair[1].id), max(pair[0].id, pair[1].id))

        selected_set = {_canon(p) for p in selected}

        entry_to_pairs: dict[int, list[tuple[OpponentEntry, OpponentEntry]]] = {}
        for pair in all_pairs:
            if _canon(pair) in selected_set:
                continue
            entry_to_pairs.setdefault(pair[0].id, []).append(pair)
            entry_to_pairs.setdefault(pair[1].id, []).append(pair)

        if self._priority_scorer is not None:
            for entry_id in entry_to_pairs:
                entry_to_pairs[entry_id] = self._priority_scorer.sort_by_priority(
                    entry_to_pairs[entry_id]
                )

        result = list(selected)
        uncovered_ids = all_entry_ids - covered_ids
        # Coverage pairs we've appended this pass; never evict them.
        protected_keys: set[tuple[int, int]] = set()

        def _find_evictable_index() -> int | None:
            """Lowest-priority pair in ``result`` whose removal keeps coverage."""
            id_count: dict[int, int] = {}
            for a, b in result:
                id_count[a.id] = id_count.get(a.id, 0) + 1
                id_count[b.id] = id_count.get(b.id, 0) + 1
            # ``result`` is priority-sorted by the caller, so last = lowest.
            for idx in range(len(result) - 1, -1, -1):
                if _canon(result[idx]) in protected_keys:
                    continue
                a, b = result[idx]
                if id_count[a.id] > 1 and id_count[b.id] > 1:
                    return idx
            return None

        while len(covered_ids) < min_covered and uncovered_ids:
            # Prefer covering entries with the fewest candidate pairs first
            # so we don't get stuck on hard-to-cover entries.
            best_entry_id = min(
                uncovered_ids,
                key=lambda eid: len(entry_to_pairs.get(eid, [])),
            )
            candidates = entry_to_pairs.get(best_entry_id, [])
            if not candidates:
                uncovered_ids.discard(best_entry_id)
                continue

            pair = candidates[0]

            if len(result) >= round_size:
                evict_idx = _find_evictable_index()
                if evict_idx is not None:
                    del result[evict_idx]
                # If no evictable pair exists every selected pair uniquely
                # covers an entry; appending still grows the round but keeps
                # the coverage contract intact.

            result.append(pair)
            protected_keys.add(_canon(pair))
            covered_ids.add(pair[0].id)
            covered_ids.add(pair[1].id)
            uncovered_ids.discard(pair[0].id)
            uncovered_ids.discard(pair[1].id)

            pair_key = _canon(pair)
            for entry_id in entry_to_pairs:
                entry_to_pairs[entry_id] = [
                    p for p in entry_to_pairs[entry_id]
                    if _canon(p) != pair_key
                ]

        return result

    def effective_ratios(
        self, entries_by_role: dict[Role, list[OpponentEntry]],
    ) -> dict[Role, float]:
        raw = {
            Role.DYNAMIC: self.config.learner_dynamic_ratio,
            Role.FRONTIER_STATIC: self.config.learner_frontier_ratio,
            Role.RECENT_FIXED: self.config.learner_recent_ratio,
        }
        non_empty = {r: w for r, w in raw.items() if entries_by_role.get(r)}
        if not non_empty:
            return {r: 0.0 for r in raw}

        # Challenge threshold: halve weight for tiers the learner dominates.
        # This redirects training time toward tiers that still provide
        # useful gradient signal, countering the "success to the successful"
        # dynamic where the learner gets ever-easier opponents.
        threshold = self.config.challenge_threshold
        for role in list(non_empty):
            wr = self.tier_win_rate(role)
            if wr is not None and wr > threshold:
                non_empty[role] *= 0.5

        total = sum(non_empty.values())
        if total <= 0:
            # All populated tiers have zero (or negative) weight — no valid
            # sampling distribution. Return all zeros so sample_for_learner
            # raises a clear error instead of dividing by zero here.
            return {r: 0.0 for r in raw}
        result = {}
        for role in raw:
            if role in non_empty:
                result[role] = non_empty[role] / total
            else:
                result[role] = 0.0
        return result
