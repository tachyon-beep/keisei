"""MatchScheduler -- role-weighted opponent selection for the tiered pool."""

from __future__ import annotations

import random
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


class MatchScheduler:
    def __init__(
        self,
        config: MatchSchedulerConfig,
        priority_scorer: PriorityScorer | None = None,
    ) -> None:
        self.config = config
        self._priority_scorer = priority_scorer

    @property
    def priority_scorer(self) -> PriorityScorer | None:
        """The PriorityScorer instance, or None if not configured."""
        return self._priority_scorer

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
        present_classes = {mc for mc in buckets if MATCH_CLASS_WEIGHTS.get(mc, 0) > 0}
        if not present_classes:
            # No weighted classes available — fall back to shuffled full round
            random.shuffle(all_pairs)
            return all_pairs[:round_size]

        total_weight = sum(MATCH_CLASS_WEIGHTS[mc] for mc in present_classes)

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
            share = max(1, round(round_size * MATCH_CLASS_WEIGHTS[mc] / total_weight))
            result.extend(pool[:share])

        # Final priority sort across all selected pairs
        if self._priority_scorer is not None:
            result = self._priority_scorer.sort_by_priority(result)
        else:
            random.shuffle(result)
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
        total = sum(non_empty.values())
        result = {}
        for role in raw:
            if role in non_empty:
                result[role] = non_empty[role] / total
            else:
                result[role] = 0.0
        return result
