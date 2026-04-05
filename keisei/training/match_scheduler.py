"""MatchScheduler -- role-weighted opponent selection for the tiered pool."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from keisei.config import MatchSchedulerConfig
from keisei.training.opponent_store import OpponentEntry, Role

if TYPE_CHECKING:
    from keisei.training.priority_scorer import PriorityScorer


class MatchScheduler:
    def __init__(
        self,
        config: MatchSchedulerConfig,
        priority_scorer: PriorityScorer | None = None,
    ) -> None:
        self.config = config
        self._priority_scorer = priority_scorer

    def sample_for_learner(
        self, entries_by_role: dict[Role, list[OpponentEntry]],
    ) -> OpponentEntry:
        ratios = self.effective_ratios(entries_by_role)
        non_empty = {r: w for r, w in ratios.items() if w > 0}
        if not non_empty:
            raise ValueError("No entries available in any tier")
        roles = list(non_empty.keys())
        weights = [non_empty[r] for r in roles]
        chosen_role = random.choices(roles, weights=weights, k=1)[0]
        return random.choice(entries_by_role[chosen_role])

    def generate_round(
        self, entries: list[OpponentEntry],
    ) -> list[tuple[OpponentEntry, OpponentEntry]]:
        """Generate all N*(N-1)/2 pairings for a full round-robin round.

        If a PriorityScorer is configured, returns pairings sorted by priority
        (highest first). Otherwise, returns shuffled pairings.
        """
        pairings: list[tuple[OpponentEntry, OpponentEntry]] = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                pairings.append((entries[i], entries[j]))
        if self._priority_scorer is not None:
            return self._priority_scorer.score_round(pairings)
        random.shuffle(pairings)
        return pairings

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
