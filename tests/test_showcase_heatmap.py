"""Unit tests for keisei.showcase.heatmap.build_heatmap()."""
from __future__ import annotations

import pytest

from keisei.showcase.heatmap import build_heatmap


def test_board_move_filters_to_same_from_square() -> None:
    """Chosen move '7g7f' → only candidates whose USI starts with '7g' are kept."""
    legal = [
        (10, "7g7f"),
        (11, "7g7f+"),
        (20, "2h2c"),       # different from-square — excluded
        (30, "P*5e"),       # drop — excluded for board moves
    ]
    probs = {10: 0.50, 11: 0.05, 20: 0.30, 30: 0.15}
    out = build_heatmap(chosen_usi="7g7f", legal_with_usi=legal, probs=probs)
    assert out == {"7g7f": pytest.approx(0.50), "7g7f+": pytest.approx(0.05)}


def test_drop_move_filters_to_same_drop_prefix() -> None:
    """Chosen move 'P*5e' → only candidates whose USI starts with 'P*' are kept."""
    legal = [
        (1, "P*5e"),
        (2, "P*4d"),
        (3, "L*3c"),       # different piece type — excluded
        (4, "7g7f"),       # board move — excluded
    ]
    probs = {1: 0.40, 2: 0.30, 3: 0.20, 4: 0.10}
    out = build_heatmap(chosen_usi="P*5e", legal_with_usi=legal, probs=probs)
    assert out == {"P*5e": pytest.approx(0.40), "P*4d": pytest.approx(0.30)}


def test_chosen_usi_is_included_in_output() -> None:
    """The chosen move itself should appear in the heatmap (so the to-square is shaded)."""
    legal = [(10, "7g7f")]
    probs = {10: 1.0}
    out = build_heatmap(chosen_usi="7g7f", legal_with_usi=legal, probs=probs)
    assert "7g7f" in out


def test_zero_probability_entries_are_omitted() -> None:
    """Entries with prob == 0.0 (legal but masked) are dropped to keep payload lean."""
    legal = [(10, "7g7f"), (11, "7g7e")]
    probs = {10: 0.95, 11: 0.0}
    out = build_heatmap(chosen_usi="7g7f", legal_with_usi=legal, probs=probs)
    assert out == {"7g7f": pytest.approx(0.95)}


def test_nan_and_inf_probabilities_are_omitted() -> None:
    """NaN/Inf probabilities (poisoned logits) must not produce invalid JSON."""
    import json
    import math

    legal = [(10, "7g7f"), (11, "7g7e"), (12, "7g7d"), (13, "7g7c")]
    probs = {10: 0.50, 11: float("nan"), 12: float("inf"), 13: float("-inf")}
    out = build_heatmap(chosen_usi="7g7f", legal_with_usi=legal, probs=probs)
    # Only the finite, positive probability survives.
    assert out == {"7g7f": pytest.approx(0.50)}
    # And the result must serialize to strict JSON (no NaN / Infinity literals).
    serialized = json.dumps(out)
    parsed = json.loads(serialized)  # would raise on NaN/Infinity literal
    assert all(math.isfinite(v) for v in parsed.values())


def test_missing_action_index_in_probs_is_skipped() -> None:
    """Defensive: legal moves whose index isn't in probs are silently skipped."""
    legal = [(10, "7g7f"), (99, "7g7e")]
    probs = {10: 0.95}  # 99 missing
    out = build_heatmap(chosen_usi="7g7f", legal_with_usi=legal, probs=probs)
    assert out == {"7g7f": pytest.approx(0.95)}
