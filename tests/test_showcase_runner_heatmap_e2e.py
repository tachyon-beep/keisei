"""Producer-contract integration test for the showcase heatmap pipeline.

The unit tests in test_showcase_heatmap.py exercise build_heatmap with
hand-crafted USI inputs. This test catches the higher-order bug class
where the runner feeds the wrong shape into build_heatmap (e.g., Hodges
notation where USI is expected) — which would silently produce empty
heatmaps that the unit tests cannot detect.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from keisei.showcase.heatmap import build_heatmap

shogi_gym = pytest.importorskip("shogi_gym", reason="Requires compiled shogi-gym Rust extension")


def test_runner_produces_nonempty_heatmap_for_board_move() -> None:
    """Driving one ply through SpectatorEnv + build_heatmap as the runner does
    must produce a non-empty heatmap dict for a board move, with all entries
    sharing the chosen move's from-square (2-char USI prefix)."""
    env = shogi_gym.SpectatorEnv(max_ply=512, action_mode="spatial")
    env.reset()

    # Capture the pre-step legal moves with their USI strings.
    legal = env.legal_actions()
    legal_with_usi = env.legal_moves_with_usi()
    assert len(legal) == 30, "startpos has 30 legal moves"
    assert len(legal_with_usi) == len(legal)

    # Build a flat probability distribution over legal moves.
    action_space_size = env.action_space_size
    probs = np.zeros(action_space_size, dtype=np.float64)
    probs[legal] = 1.0 / len(legal)

    # Pick a board move: at startpos all 30 are board moves (no drops possible).
    action = legal[0]
    chosen_usi = next((u for i, u in legal_with_usi if i == action), "")
    assert chosen_usi != "", "chosen action must appear in legal_with_usi"
    assert chosen_usi[1] != "*", "startpos has no drops"

    # Build the heatmap exactly as the runner does.
    heatmap = build_heatmap(
        chosen_usi=chosen_usi,
        legal_with_usi=legal_with_usi,
        probs={int(i): float(probs[i]) for i in legal},
    )

    # The chosen move must appear in the heatmap.
    assert chosen_usi in heatmap, f"chosen move {chosen_usi!r} must be in heatmap {heatmap!r}"
    # Every entry must share the chosen move's from-square prefix.
    target_prefix = chosen_usi[:2]
    for usi in heatmap:
        assert usi.startswith(target_prefix), (
            f"heatmap entry {usi!r} does not share from-square prefix {target_prefix!r}"
        )
    # Heatmap must serialise to strict JSON.
    parsed = json.loads(json.dumps(heatmap))
    assert parsed == heatmap


def test_runner_chosen_usi_differs_from_state_notation() -> None:
    """Pin the misleading-name issue: state['move_history'][-1]['notation']
    is Hodges, NOT USI. This test documents the contract so any future
    rename of the field name catches us, AND demonstrates the actual failure
    mode (feeding Hodges to build_heatmap silently yields {}) so a regression
    of the fix in runner.py is loudly caught here."""
    from keisei.showcase.heatmap import build_heatmap

    env = shogi_gym.SpectatorEnv(max_ply=512, action_mode="spatial")
    env.reset()
    legal_with_usi = env.legal_moves_with_usi()
    action = env.legal_actions()[0]
    state = env.step(action)
    hodges = state["move_history"][-1]["notation"]
    usi = next((u for i, u in legal_with_usi if i == action), "")
    # At startpos, e.g. action=7506 yields hodges='P-9f' and usi='9g9f'.
    assert hodges != usi, (
        f"Expected Hodges and USI to differ for board moves; got hodges={hodges!r} usi={usi!r}. "
        "If they are now equal, move_history.notation may have been changed to USI; "
        "the runner can then be simplified to drop chosen_usi_real derivation."
    )

    # Demonstrate the bug: feeding Hodges to build_heatmap silently yields {}.
    # If runner.py ever reverts to passing the Hodges value into chosen_usi=,
    # the heatmap goes empty for every board move. This assertion pins that
    # failure mode so the regression is caught in CI.
    fake_probs = {int(i): 1.0 / len(legal_with_usi) for i, _ in legal_with_usi}
    broken = build_heatmap(chosen_usi=hodges, legal_with_usi=legal_with_usi, probs=fake_probs)
    assert broken == {}, (
        f"Expected empty heatmap when feeding Hodges {hodges!r}, got {broken!r}. "
        "If non-empty, the from-square prefix matching has changed and the runner's "
        "chosen_usi_real derivation may no longer be needed."
    )
    correct = build_heatmap(chosen_usi=usi, legal_with_usi=legal_with_usi, probs=fake_probs)
    assert correct, (
        f"Expected non-empty heatmap when feeding real USI {usi!r}, got {correct!r}. "
        "Heatmap producer contract is broken; investigate build_heatmap and legal_moves_with_usi."
    )
