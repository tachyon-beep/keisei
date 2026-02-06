# Analysis: keisei/shogi/features.py

**Lines:** 193
**Role:** Defines the `FeatureSpec` registry and observation tensor builders for converting Shogi game state into neural network input. Provides `build_core46` (46-plane observation) and `build_core46_all` (51-plane extended observation), plus optional feature plane functions (check, repetition, promotion zone, last-2-ply, hand one-hot). The `FEATURE_SPECS` dictionary is used by `model_manager.py` to determine observation shape.
**Key dependencies:** Imports `MOVE_COUNT_NORMALIZATION_FACTOR` from `keisei.constants`. Accessed by `keisei.training.model_manager` (for `FEATURE_SPECS` / `num_planes`). The `build_core46` function expects a game object with specific attributes.
**Analysis depth:** FULL

## Summary

This file contains multiple critical issues that would cause runtime failures if the `build_core46` or `build_core46_all` functions were called with a real `ShogiGame` instance. Fortunately, these builders are currently dead code in production -- `model_manager.py` only reads `num_planes` from `FEATURE_SPECS` and never calls `build()`. The actual observation generation uses `shogi_game_io.generate_neural_network_observation()`. However, the code presents itself as the canonical feature builder and would silently produce incorrect observations even if the attribute errors were fixed, due to multiple semantic divergences from the production observation generator. This is a significant maintenance trap.

## Critical Findings

### [60-113] build_core46 is incompatible with ShogiGame and would crash at runtime
**What:** `build_core46` accesses the game object via attributes that do not exist on `ShogiGame`:
- `game.OBS_CURR_PLAYER_UNPROMOTED_START` (lines 72, 77) -- `ShogiGame` has no such attribute; this is a module-level constant in `shogi_core_definitions.py`
- `game.OBS_UNPROMOTED_ORDER` (lines 73, 78, 94) -- same; module-level constant, not an instance attribute
- `game.OBS_PROMOTED_ORDER` (lines 85, 90) -- same
- `game.OBS_CURR_PLAYER_PROMOTED_START` (line 84) -- same
- `game.OBS_OPP_PLAYER_UNPROMOTED_START` (line 77) -- same
- `game.OBS_OPP_PLAYER_PROMOTED_START` (line 89) -- same
- `game.OBS_CURR_PLAYER_HAND_START` (line 95) -- same
- `game.OBS_OPP_PLAYER_HAND_START` (line 103) -- same
- `game.OBS_CURR_PLAYER_INDICATOR` (line 105) -- same
- `game.OBS_MOVE_COUNT` (line 108) -- same
- `game.OBS_RESERVED_1` (line 111) -- same
- `game.OBS_RESERVED_2` (line 112) -- same
- `game.Color.BLACK` (line 106) -- `ShogiGame` does not expose `Color` as a class/instance attribute
- `piece.piece_type` (lines 73, 78, 85, 90) -- `Piece` class uses `piece.type`, not `piece.piece_type`
- `piece.is_promoted()` (line 69) -- `Piece.is_promoted` is a boolean attribute, not a callable method

Calling `build_core46(game)` with a real `ShogiGame` instance would raise `AttributeError` at line 72 on the very first piece encountered.

**Why it matters:** If any future developer or feature work attempts to use the `FeatureSpec.build()` method (which delegates to `build_core46`) with a real game, it will crash immediately. The function's docstring claims to "mirror `generate_neural_network_observation` in `shogi_game_io.py`" (line 58), which is false. The test suite masks this by using a `DummyGame` stub that manually sets all these attributes with different types (strings instead of `PieceType` enums, ints instead of `Color` enums).

**Evidence:**
```python
# features.py line 72-73:
idx = game.OBS_CURR_PLAYER_UNPROMOTED_START + game.OBS_UNPROMOTED_ORDER.index(piece.piece_type)

# But ShogiGame has no OBS_* attributes. These are module-level constants.
# And Piece uses .type, not .piece_type
```

### [95-103] Hand piece access uses wrong key type for ShogiGame.hands
**What:** `game.hands[game.current_player]` uses the `Color` enum directly as a dictionary key. In the real `ShogiGame`, `self.hands` is keyed by `Color.BLACK.value` (int 0) and `Color.WHITE.value` (int 1), not by `Color` enum instances.
**Why it matters:** This would raise `KeyError` at runtime with a real `ShogiGame` instance, even if the other attribute errors were fixed.
**Evidence:**
```python
# features.py line 95-96:
obs[game.OBS_CURR_PLAYER_HAND_START + i, :, :] = game.hands[game.current_player].get(pt, 0)

# shogi_game.py line 116-119 (reset method):
self.hands = {
    Color.BLACK.value: {ptype: 0 for ptype in get_unpromoted_types()},
    Color.WHITE.value: {ptype: 0 for ptype in get_unpromoted_types()},
}
```

### [95-103] Hand piece iteration uses game.OBS_UNPROMOTED_ORDER which includes KING
**What:** The hand piece loop iterates over `game.OBS_UNPROMOTED_ORDER`, which (in the real definitions) includes `PieceType.KING` as its 8th element. But hands do not contain kings. This means hand channel indices 28+7=35 (which is `OBS_OPP_PLAYER_HAND_START`) would be written to by the current player's king hand count (always 0, so no corruption, but the logic is semantically wrong). The hand channels should iterate over the 7 droppable types, not the 8 unpromoted board types.
**Why it matters:** In the current state this is masked by kings always having 0 count, but the off-by-one in the conceptual mapping means the 8th iteration writes to channel 35, which is the start of the opponent's hand block. If a king count were ever non-zero (impossible in valid Shogi, but indicates fragile reasoning), data corruption would occur.
**Evidence:**
```python
# OBS_UNPROMOTED_ORDER has 8 elements (P, L, N, S, G, B, R, K)
# But hand channels are 7 per player (28-34 and 35-41)
# Iterating i from 0..7 with OBS_CURR_PLAYER_HAND_START (28) + i gives channels 28-35
# Channel 35 is OBS_OPP_PLAYER_HAND_START -- overlap!
```

## Warnings

### [50-113] build_core46 diverges semantically from the production observation generator
**What:** Even if the attribute access issues were fixed, `build_core46` produces different observations than `shogi_game_io.generate_neural_network_observation`:

1. **No perspective flipping**: `build_core46` uses raw `(r, c)` coordinates (line 92). The production code flips coordinates for White's perspective (`flipped_r = 8 - r, flipped_c = 8 - c` when not Black).

2. **Different move count normalization**: `build_core46` divides by `MOVE_COUNT_NORMALIZATION_FACTOR` (512.0, line 109). The production code divides by `game.max_moves_per_game` (default 500.0, line 536 of shogi_game_io.py).

3. **Different hand piece normalization**: `build_core46` stores raw hand counts (line 95-103). The production code normalizes by dividing by 18.0 (line 514 of shogi_game_io.py).

4. **Different hand piece ordering**: `build_core46` uses `game.OBS_UNPROMOTED_ORDER` (8 types including King). The production code uses `get_unpromoted_types()` (7 types, no King).

**Why it matters:** If `build_core46` were ever activated as the observation builder (e.g., by wiring `FeatureSpec.build()` into the training loop), all training data would be silently corrupted. The neural network would receive observations with different semantics than intended, leading to failed training with no error messages. This is the most dangerous class of bug in ML systems.

### [98-102] Fragile opponent color computation
**What:** The opponent color is computed with a fallback pattern:
```python
opp = (game.current_player.opponent() if hasattr(game.current_player, "opponent") else (1 - game.current_player))
```
This dual-path logic suggests uncertainty about whether `current_player` is a `Color` enum (which has `.opponent()`) or a plain int. The production `ShogiGame` uses `Color` enum, so the `hasattr` check will always be True. The fallback `(1 - game.current_player)` would fail on a `Color` enum anyway (you can't subtract an enum from an int).
**Why it matters:** The defensive coding pattern here is a code smell indicating the author was unsure about the game object's contract. It would mask type errors in test stubs.

### [119-163] Optional feature plane functions have inconsistent game object expectations
**What:** The optional plane functions (`add_check_plane`, `add_prom_zone_plane`, etc.) also access `game.Color.BLACK` (line 141), which is not a valid attribute path on `ShogiGame`. They use `hasattr` guards for some game attributes (`is_in_check`, `repetition_count`, `is_sennichite`, `move_history`) but not for `game.Color`. These functions are called by `build_core46_all` which inherits all the same incompatibilities.

### [149-153] add_last2ply_plane accesses move_history items as objects with to_square attribute
**What:** The function checks `hasattr(move, "to_square")` on each item in `game.move_history`. But in `ShogiGame`, `move_history` is a list of dictionaries (not objects with attributes). The dict entries have a `"move"` key containing a `MoveTuple`, not a `to_square` attribute. So this function would silently produce an all-zeros plane for a real game.
**Why it matters:** The `last2ply` feature would be silently disabled, providing no useful signal to the neural network if this code path were used.
**Evidence:**
```python
# features.py line 150-153:
for move in game.move_history[-2:]:
    if hasattr(move, "to_square"):
        r, c = move.to_square

# ShogiGame.move_history entries are dicts like:
# {"move": (from_r, from_c, to_r, to_c, promote), "is_drop": False, ...}
```

### [182-185] Test/dummy FeatureSpec entries pollute the production registry
**What:** `DUMMY_FEATS_SPEC`, `TEST_FEATS_SPEC`, and `RESUME_FEATS_SPEC` are defined at module level and added to `FEATURE_SPECS`. These use `build_core46` as their builder, meaning they inherit all the broken behavior. More importantly, test fixture data should not be in production code.
**Why it matters:** If configuration accidentally references `"dummyfeats"` or `"testfeats"` in production, it would silently use the broken `build_core46` builder. Test-only data in production modules increases the risk of accidental misuse and confuses the API surface.

## Observations

### [12] FEATURE_REGISTRY is a mutable module-level global
**What:** `FEATURE_REGISTRY: Dict[str, Callable] = {}` is populated at import time by the `@register_feature` decorator. It is never used anywhere in the codebase other than in test assertions (`test_features.py` lines 136-137).
**Why it matters:** The registry pattern suggests a plugin architecture that was designed but never completed. The `FEATURE_SPECS` dict (lines 187-193) serves the same lookup purpose and is the one actually used by `model_manager.py`. The registry is dead code.

### [23-34] FeatureSpec class is underutilized
**What:** `FeatureSpec` stores a name, builder, and num_planes. The `build()` method is never called in production. Only `num_planes` is used (by `model_manager.py` to determine observation shape).
**Why it matters:** The class adds indirection without value. A simple dict of `{name: num_planes}` would suffice for the current usage pattern. However, the class is positioned for future extensibility, which is a reasonable design choice if the builders are fixed.

### [38-44] EXTRA_PLANES dict uses string keys
**What:** The extra plane offsets are stored with string keys like `"check"`, `"repetition"`, etc. This is fine for a small fixed set but is not type-safe.

## Verdict
**Status:** CRITICAL
**Recommended action:** This file requires significant remediation:
1. **Immediate**: Add prominent docstring warnings that `build_core46` and `build_core46_all` are NOT compatible with `ShogiGame` and are not used in production. This prevents future developers from accidentally wiring them in.
2. **Short-term**: Either fix `build_core46` to work with real `ShogiGame` instances (matching the semantics of `shogi_game_io.generate_neural_network_observation`) or delete the builder functions entirely and keep only the `FeatureSpec` entries with `num_planes` metadata.
3. **Short-term**: Move test/dummy `FeatureSpec` entries out of production code and into test fixtures.
4. **Medium-term**: Consolidate the observation generation to a single code path. Having two implementations (`features.py` builders and `shogi_game_io.generate_neural_network_observation`) that claim to do the same thing but diverge is a maintenance time bomb.
**Confidence:** HIGH -- The critical findings are verified by cross-referencing the `ShogiGame` class definition, `Piece` class attributes, and the `shogi_game_io.py` production implementation. The test suite uses a `DummyGame` stub that masks all these issues.
