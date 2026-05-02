"""Build the policy-preference heatmap for a showcase ply.

The heatmap is a {usi: probability} dict containing legal moves that share
the chosen move's from-square (board moves) or piece type (drops). It is
serialized to JSON and stored in showcase_moves.move_heatmap_json.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _move_prefix(usi: str) -> str:
    """First two chars of a USI move, identifying its from-square or drop prefix.

    Examples: '7g7f' -> '7g'; '7g7f+' -> '7g'; 'P*5e' -> 'P*'.
    """
    return usi[:2]


def build_heatmap(
    *,
    chosen_usi: str,
    legal_with_usi: Sequence[tuple[int, str]],
    probs: Mapping[int, float],
) -> dict[str, float]:
    """Filter legal moves to those sharing the chosen move's from-square (or drop
    prefix) and pair each with its policy probability.

    Args:
        chosen_usi: The USI string of the move that was actually played this ply.
        legal_with_usi: All legal (action_index, usi_string) pairs at this position
            (typically from SpectatorEnv.legal_moves_with_usi()).
        probs: Full softmax-over-legal-moves distribution, keyed by action index.

    Returns:
        A {usi: probability} dict suitable for json.dumps() and storage.
        Entries with probability 0.0 or missing from `probs` are omitted.
    """
    target = _move_prefix(chosen_usi)
    out: dict[str, float] = {}
    for idx, usi in legal_with_usi:
        if _move_prefix(usi) != target:
            continue
        prob = probs.get(idx)
        if prob is None or not math.isfinite(prob) or prob <= 0.0:
            continue
        out[usi] = float(prob)
    return out
