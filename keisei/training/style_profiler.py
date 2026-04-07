"""StyleProfiler: aggregate game features into checkpoint style profiles.

Reads per-game feature rows from the game_features table, computes
league-relative metrics and percentiles, assigns rule-based style labels,
and generates commentary facts.  Writes results to the style_profiles table.

Designed to run as a lightweight batch job after tournament rounds.
"""

from __future__ import annotations

import bisect
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from keisei.db import (
    read_all_game_features,
    write_style_profile,
)

logger = logging.getLogger(__name__)

# Sample thresholds (§7)
THRESHOLD_INSUFFICIENT = 25
THRESHOLD_PROVISIONAL = 75
THRESHOLD_TREND = 200


def _percentile_rank(value: float, sorted_values: list[float]) -> float:
    """Compute the percentile rank of a value within a sorted list.

    Uses bisect for O(log n) instead of linear scan.
    Returns a value in [0, 100].
    """
    if not sorted_values:
        return 50.0
    n = len(sorted_values)
    count_below = bisect.bisect_left(sorted_values, value)
    count_equal = bisect.bisect_right(sorted_values, value) - count_below
    return ((count_below + 0.5 * count_equal) / n) * 100


def _safe_mean(values: list[float]) -> float | None:
    """Compute mean, returning None if empty."""
    if not values:
        return None
    return sum(values) / len(values)


def _mode(values: list[Any]) -> Any | None:
    """Return the most common value, or None if empty."""
    if not values:
        return None
    counter = Counter(values)
    return counter.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Style label rules (§10.1)
# ---------------------------------------------------------------------------

_STYLE_RULES: list[tuple[str, dict[str, tuple[str, float]]]] = [
    # (label, {metric: (comparator, threshold_percentile)})
    ("Sharp tactical opener", {
        "first_capture_ply_mean": ("<=", 30),
        "avg_game_length": ("<=", 45),
    }),
    ("Patient attacker", {
        "avg_game_length": (">=", 65),
        "num_captures_mean": (">=", 55),
    }),
    ("Drop-heavy scrapper", {
        "drops_per_game": (">=", 75),
        "num_early_drops_mean": (">=", 60),
    }),
    ("Slow builder", {
        "avg_game_length": (">=", 70),
        "first_capture_ply_mean": (">=", 60),
    }),
    ("Flexible opener", {
        "opening_diversity_index": (">=", 75),
    }),
    ("Aggressive promoter", {
        "promotions_per_game": (">=", 75),
        "first_capture_ply_mean": ("<=", 40),
    }),
    ("Chaotic brawler", {
        "avg_game_length": ("<=", 35),
        "num_captures_mean": (">=", 65),
        "drops_per_game": (">=", 55),
    }),
    ("Long-game grinder", {
        "avg_game_length": (">=", 80),
        "game_length_variance": ("<=", 40),
    }),
    ("Early rook swinger", {
        "rook_moved_early_rate": (">=", 70),
    }),
    ("Defensive builder", {
        "king_moves_early_rate": (">=", 65),
        "first_capture_ply_mean": (">=", 55),
    }),
]

# Contradictory label pairs — never assign both
_CONTRADICTIONS: list[tuple[str, str]] = [
    ("Sharp tactical opener", "Slow builder"),
    ("Sharp tactical opener", "Patient attacker"),
    ("Chaotic brawler", "Slow builder"),
    ("Chaotic brawler", "Long-game grinder"),
    ("Aggressive promoter", "Defensive builder"),
]


def _check_rule(
    conditions: dict[str, tuple[str, float]],
    percentiles: dict[str, float],
) -> bool:
    """Check if all conditions in a style rule are met."""
    for metric, (comparator, threshold) in conditions.items():
        pct = percentiles.get(metric)
        if pct is None:
            return False
        if comparator == ">=" and pct < threshold:
            return False
        if comparator == "<=" and pct > threshold:
            return False
    return True


def _assign_labels(
    percentiles: dict[str, float],
) -> tuple[str | None, list[str]]:
    """Assign primary style + up to 2 secondary traits.

    Returns (primary_style, secondary_traits).
    """
    matching: list[str] = []
    for label, conditions in _STYLE_RULES:
        if _check_rule(conditions, percentiles):
            matching.append(label)

    if not matching:
        return None, []

    # Filter contradictions
    filtered: list[str] = []
    for label in matching:
        contradicts = False
        for a, b in _CONTRADICTIONS:
            if label == a and b in filtered:
                contradicts = True
                break
            if label == b and a in filtered:
                contradicts = True
                break
        if not contradicts:
            filtered.append(label)
        if len(filtered) >= 3:
            break

    primary = filtered[0] if filtered else None
    secondary = filtered[1:3]
    return primary, secondary


# ---------------------------------------------------------------------------
# Commentary generation (§11)
# ---------------------------------------------------------------------------

_COMMENTARY_TEMPLATES: list[tuple[str, str, str, str, float]] = [
    # (category, metric, comparator, template, threshold_percentile)
    # template uses {value} and {pct} placeholders
    ("opening", "preferred_first_move_black", "mode_freq>=", "Usually opens with action {value} as Black", 40),
    ("opening", "preferred_first_move_white", "mode_freq>=", "Usually responds with action {value} as White", 40),
    ("tempo", "first_capture_ply_mean", "pct<=", "Starts exchanging earlier than {pct}% of the league", 30),
    ("tempo", "first_capture_ply_mean", "pct>=", "Takes longer to start exchanging than {pct}% of the league", 75),
    ("game_length", "avg_game_length", "pct<=", "Shorter games than {pct}% of the league", 35),
    ("game_length", "avg_game_length", "pct>=", "Longer games than {pct}% of the league", 70),
    ("tactical", "drops_per_game", "pct>=", "Uses drops more than most league rivals", 70),
    ("tactical", "promotions_per_game", "pct>=", "Promotes aggressively", 75),
    ("tactical", "num_captures_mean", "pct>=", "Captures more pieces than most rivals", 70),
    ("tactical", "num_early_drops_mean", "pct>=", "Drops pieces early more often than average", 65),
    ("positional", "rook_moved_early_rate", "pct>=", "Most likely to swing the rook early", 75),
    ("positional", "rook_moved_early_rate", "pct<=", "Rarely shifts the rook in the opening", 25),
    ("positional", "king_moves_early_rate", "pct>=", "Develops the king early", 70),
    ("positional", "opening_diversity_index", "pct>=", "Plays a wide variety of openings", 75),
    ("positional", "opening_diversity_index", "pct<=", "Sticks to a narrow set of openings", 25),
]


def _generate_commentary(
    raw_metrics: dict[str, Any],
    percentiles: dict[str, float],
) -> list[dict[str, Any]]:
    """Generate ranked commentary facts from metrics and percentiles."""
    facts: list[dict[str, Any]] = []

    for category, metric, condition, template, threshold in _COMMENTARY_TEMPLATES:
        pct = percentiles.get(metric)
        value = raw_metrics.get(metric)

        if condition.startswith("mode_freq>="):
            # Mode-frequency templates use categorical metrics (e.g. preferred
            # first move) that don't have percentiles.  Check frequency directly.
            if value is None:
                continue
            freq = raw_metrics.get(f"{metric}_freq", 0)
            if freq >= threshold / 100.0:
                confidence = "high" if freq >= 0.5 else "medium"
                text = template.format(value=value, pct=0)
                facts.append({"text": text, "category": category, "confidence": confidence})
        elif pct is None or value is None:
            continue
        elif condition == "pct<=" and pct <= threshold:
            confidence = "high" if pct <= threshold - 10 else "medium"
            text = template.format(value=value, pct=round(100 - pct))
            facts.append({"text": text, "category": category, "confidence": confidence})
        elif condition == "pct>=" and pct >= threshold:
            confidence = "high" if pct >= threshold + 10 else "medium"
            text = template.format(value=value, pct=round(pct))
            facts.append({"text": text, "category": category, "confidence": confidence})

    # Sort: high confidence first, then by category diversity
    facts.sort(key=lambda f: (0 if f["confidence"] == "high" else 1, f["category"]))

    # Cap at 5 facts, avoid repeating categories
    selected: list[dict[str, Any]] = []
    for fact in facts:
        if len(selected) >= 5:
            break
        # Allow up to 2 from same category
        cat_count = sum(1 for s in selected if s["category"] == fact["category"])
        if cat_count < 2:
            selected.append(fact)

    return selected


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------


class StyleProfiler:
    """Aggregate game features into checkpoint style profiles."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def recompute_all(self) -> int:
        """Recompute profiles for all checkpoints with game data.

        Returns the number of profiles written.

        Note: reads all game_features rows (O(N) full scan).  Acceptable for
        v1 where the table is small; for long training runs, consider
        incremental aggregation or epoch-windowed reads.
        """
        all_features = read_all_game_features(self.db_path)
        if not all_features:
            return 0

        # Group by checkpoint_id
        by_checkpoint: dict[int, list[dict[str, Any]]] = {}
        for row in all_features:
            cid = row["checkpoint_id"]
            by_checkpoint.setdefault(cid, []).append(row)

        # Compute raw metrics for each checkpoint
        all_metrics: dict[int, dict[str, Any]] = {}
        for cid, rows in by_checkpoint.items():
            metrics = self._aggregate_features(rows)
            if metrics is not None:
                all_metrics[cid] = metrics

        if not all_metrics:
            return 0

        # Compute league-wide percentiles
        all_percentiles = self._compute_percentiles(all_metrics)

        # Classify and write profiles
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        written = 0
        for cid, metrics in all_metrics.items():
            percentiles = all_percentiles.get(cid, {})
            game_count = len(by_checkpoint[cid])
            status = self._profile_status(game_count)

            if status == "insufficient":
                primary_style = None
                secondary_traits: list[str] = []
                commentary: list[dict[str, Any]] = []
            else:
                primary_style, secondary_traits = _assign_labels(percentiles)
                commentary = _generate_commentary(metrics, percentiles)

            write_style_profile(self.db_path, {
                "checkpoint_id": cid,
                "recomputed_at": now,
                "profile_status": status,
                "games_sampled": game_count,
                "raw_metrics": metrics,
                "percentiles": percentiles,
                "primary_style": primary_style,
                "secondary_traits": secondary_traits,
                "commentary": commentary,
            })
            written += 1

        logger.info(
            "Style profiles recomputed: %d checkpoints (%d total game rows)",
            written, len(all_features),
        )
        return written

    def _aggregate_features(
        self, rows: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Compute raw metric values from per-game feature rows."""
        if not rows:
            return None

        game_lengths = [r["total_plies"] for r in rows]
        first_captures = [r["first_capture_ply"] for r in rows if r["first_capture_ply"] is not None]
        first_drops = [r["first_drop_ply"] for r in rows if r["first_drop_ply"] is not None]
        drops = [r["num_drops"] for r in rows]
        promotions = [r["num_promotions"] for r in rows]
        captures = [r["num_captures"] for r in rows]
        early_drops = [r["num_early_drops"] for r in rows]
        rook_moves_20 = [r["rook_moves_in_20"] for r in rows]
        king_moves_30 = [r["king_moves_in_30"] for r in rows]
        king_disp_20 = [r["king_displacement_20"] for r in rows]

        # Side-specific
        black_rows = [r for r in rows if r["side"] == "black"]
        white_rows = [r for r in rows if r["side"] == "white"]
        black_first_actions = [r["first_action"] for r in black_rows if r["first_action"] is not None]
        white_first_actions = [r["first_action"] for r in white_rows if r["first_action"] is not None]

        # Rook moved early rate
        rook_early = [1 for r in rows if r["rook_moved_ply"] is not None and r["rook_moved_ply"] <= 20]
        rook_moved_early_rate = len(rook_early) / len(rows) if rows else 0.0

        # King moves early rate
        king_early_moves = [r["king_moves_in_30"] for r in rows]
        king_moves_early_rate = _safe_mean([1.0 if k > 0 else 0.0 for k in king_early_moves]) or 0.0

        # Opening diversity
        opening_seqs = [r["opening_seq_3"] for r in rows if r["opening_seq_3"] is not None]
        unique_openings = len(set(opening_seqs))
        opening_diversity_index = unique_openings / len(opening_seqs) if opening_seqs else 0.0

        # Preferred first moves
        pref_black = _mode(black_first_actions)
        pref_white = _mode(white_first_actions)
        pref_black_freq = (
            Counter(black_first_actions).most_common(1)[0][1] / len(black_first_actions)
            if black_first_actions else 0.0
        )
        pref_white_freq = (
            Counter(white_first_actions).most_common(1)[0][1] / len(white_first_actions)
            if white_first_actions else 0.0
        )

        # Game length variance
        gl_mean = _safe_mean(game_lengths)
        game_length_variance = (
            sum((g - gl_mean) ** 2 for g in game_lengths) / len(game_lengths)
            if gl_mean is not None and game_lengths else 0.0
        )

        # Win/loss/draw rates
        wins = sum(1 for r in rows if r["result"] == "win")
        losses = sum(1 for r in rows if r["result"] == "loss")
        draws_count = sum(1 for r in rows if r["result"] == "draw")
        total = len(rows)

        # Short game rate (below median game length)
        if game_lengths:
            sorted_gl = sorted(game_lengths)
            median_gl = sorted_gl[len(sorted_gl) // 2]
            short_game_rate = sum(1 for g in game_lengths if g < median_gl) / total
        else:
            short_game_rate = 0.0

        return {
            # §8.1 Opening features
            "preferred_first_move_black": pref_black,
            "preferred_first_move_black_freq": pref_black_freq,
            "preferred_first_move_white": pref_white,
            "preferred_first_move_white_freq": pref_white_freq,
            "opening_diversity_index": opening_diversity_index,
            # §8.2 Tempo and aggression
            "avg_game_length": gl_mean,
            "first_capture_ply_mean": _safe_mean(first_captures),
            "first_drop_ply_mean": _safe_mean(first_drops),
            "num_captures_mean": _safe_mean(captures),
            "short_game_rate": short_game_rate,
            # §8.3 Drop and promotion
            "drops_per_game": _safe_mean(drops),
            "promotions_per_game": _safe_mean(promotions),
            "num_early_drops_mean": _safe_mean(early_drops),
            # §8.4 Positional proxies
            "rook_moved_early_rate": rook_moved_early_rate,
            "rook_moves_in_20_mean": _safe_mean(rook_moves_20),
            "king_displacement_20_mean": _safe_mean(king_disp_20),
            "king_moves_in_30_mean": _safe_mean(king_moves_30),
            "king_moves_early_rate": king_moves_early_rate,
            # §8.5 Volatility
            "game_length_variance": game_length_variance,
            "win_rate": wins / total if total else 0.0,
            "loss_rate": losses / total if total else 0.0,
            "draw_rate": draws_count / total if total else 0.0,
        }

    def _compute_percentiles(
        self, all_metrics: dict[int, dict[str, Any]]
    ) -> dict[int, dict[str, float]]:
        """Compute league-relative percentiles for each metric across checkpoints."""
        if not all_metrics:
            return {}

        # Collect the union of all numeric metric keys across checkpoints.
        # Using a single sample would miss metrics that are None for that
        # checkpoint but present for others (e.g. first_capture_ply_mean).
        numeric_keys = sorted({
            key
            for metrics in all_metrics.values()
            for key, value in metrics.items()
            if value is not None and isinstance(value, (int, float))
        })

        # Build sorted lists per metric
        sorted_per_metric: dict[str, list[float]] = {}
        for key in numeric_keys:
            values = [
                m[key] for m in all_metrics.values()
                if m.get(key) is not None and isinstance(m[key], (int, float))
            ]
            sorted_per_metric[key] = sorted(values)

        # Compute percentiles per checkpoint
        result: dict[int, dict[str, float]] = {}
        for cid, metrics in all_metrics.items():
            pcts: dict[str, float] = {}
            for key in numeric_keys:
                value = metrics.get(key)
                if value is not None and isinstance(value, (int, float)):
                    pcts[key] = _percentile_rank(value, sorted_per_metric[key])
                else:
                    pcts[key] = 50.0
            result[cid] = pcts
        return result

    @staticmethod
    def _profile_status(game_count: int) -> str:
        """Determine profile status from game count (§7)."""
        if game_count < THRESHOLD_INSUFFICIENT:
            return "insufficient"
        if game_count < THRESHOLD_PROVISIONAL:
            return "provisional"
        return "established"
