# Showcase: Last-Move Indicator + Policy Heatmap

**Date:** 2026-05-03
**Status:** Draft
**Scope:** Webui showcase tab visual enhancement — last-move highlight (always on) and toggleable policy-preference heatmap. No changes to training, league, or evaluation paths.

## Problem

The showcase board renders the position after each move, but offers no visual indication of *which* squares were involved in the most recent move. A spectator must consult the move log and mentally translate USI notation back to board coordinates.

The board also gives no insight into how concentrated or diffuse the model's policy was — i.e., was the chosen move an obvious favourite, or did it edge out close alternatives? `top_candidates` exists in the DB but only the chosen move's USI is resolved; the other entries carry placeholder action IDs (`runner.py:184`).

## Goals

1. Highlight the from-square (subtle) and to-square (prominent) of the last move on the showcase board. For drops, only the destination is highlighted.
2. Provide a toggleable overlay that shades every square the last-moved piece *could* have legally moved to, with intensity proportional to the model's policy probability for that move. For drops, the overlay covers all legal drop squares for the dropped piece type.

## Non-Goals

- Per-move value estimates (would require a 1-step lookahead — explicitly out of scope per design discussion).
- Training-tab board enhancements (the dead `lastMoveIdx` placeholder in `App.svelte:55-61` is removed, but no new highlighting added there).
- Backfilling heatmap data for matches played before this lands (NULL handled gracefully).
- Heatmap on plies other than the most recent (no scrubbing UX added in this scope).

## Architecture

### Data flow

```
Rust SpectatorEnv
  └── new method: legal_moves_with_usi() -> [(action_idx, usi_str)]
        |
        v
Python runner.py
  └── filter to candidates sharing chosen move's from-square (or drop prefix)
  └── pair with softmax policy probability
        |
        v
SQLite showcase_moves
  └── new column: move_heatmap_json TEXT (nullable)
        |
        v
FastAPI showcase WebSocket payload (existing path, just carries the new field)
        |
        v
Svelte ShowcaseView
  ├── parseUsi() helper extracts from/to indices from current move's usi_notation
  └── parseHeatmap() helper consumes move_heatmap_json
        |
        v
Board.svelte
  ├── lastMoveFromIdx + lastMoveToIdx props (replacing lastMoveIdx)
  └── heatmap prop ({squareIdx: probability})
```

### Backend changes

**Rust (`shogi-engine/crates/shogi-gym/src/spectator.rs`):**

Add a single read-only PyO3 method:

```rust
/// Return all legal moves at the current position with their USI strings.
pub fn legal_moves_with_usi(&mut self) -> Vec<(usize, String)> {
    let perspective = self.game.position.current_player;
    let moves = self.game.legal_moves();
    moves.into_iter()
        .map(|mv| {
            let idx = self.mapper.encode(mv, perspective)
                .expect("legal move must be encodable");
            (idx, move_usi(mv))
        })
        .collect()
}
```

No state mutation. Mirrors the existing `legal_actions()` pattern (`spectator.rs:206-214`). `move_usi()` already exists at `spectator_data.rs:93`.

**Rust unit test:** at startpos, returned set equals `legal_actions()` zipped with valid USI strings, length 30.

**Python (`keisei/showcase/runner.py`):**

After computing `legal_probs` and choosing `action`, build the heatmap dict by:

1. Parsing the chosen move's USI to extract its from-prefix:
   - Board move (e.g., `"7g7f"`, `"7g7f+"`): from-prefix = first 2 chars (`"7g"`).
   - Drop (e.g., `"P*5e"`): drop-prefix = first 2 chars (`"P*"`).
2. Calling `env.legal_moves_with_usi()` (called *before* `env.step()` so the position still matches the policy distribution).
3. Filtering to entries whose USI starts with the same prefix.
4. Mapping each `action_idx` back through the `probs` array we already computed.
5. Storing as `{usi: probability}` JSON (USI keys are simpler for the frontend than action indices).

Extract this as a helper `_build_heatmap(chosen_usi, legal_with_usi, probs) -> dict[str, float]` for testability — pure function, no env coupling. Unit-test it with a hand-rolled board-move case and a drop case.

Persist via a new `move_heatmap_json` parameter on `write_showcase_move()`.

**SQLite schema (`keisei/db.py`):**

- Bump `SCHEMA_VERSION` from 6 to 7.
- Add `move_heatmap_json TEXT` to the `CREATE TABLE showcase_moves` DDL (for fresh databases).
- Add `_migrate_v6_to_v7()` that calls `_migrate_add_column(conn, "showcase_moves", "move_heatmap_json", "TEXT")`.
- Register in `MIGRATIONS` dict.
- Update `write_showcase_move()` insert statement and signature to accept and store the new column.

Old rows have NULL → frontend renders no heatmap (toggle has no visible effect on those plies). Acceptable; pre-existing matches don't need backfill.

**FastAPI server:** the showcase WebSocket payload is built from `showcase_moves` row dicts. Verify the new column flows through automatically (it should — most code uses `dict(row)` patterns) and add it to any explicit field projections if present.

### Frontend changes

**New helper `webui/src/lib/usiCoords.js`:**

```javascript
/**
 * Parse a USI move string and return board indices (0-80, row*9+col) for
 * from-square and to-square. For drops, fromIdx is null.
 *
 * USI examples:
 *   "7g7f"   -> board move, from "7g" to "7f"
 *   "7g7f+"  -> board move with promotion
 *   "P*5e"   -> pawn drop to "5e"
 *
 * Square notation: file 1-9 (right-to-left from black's POV), rank a-i (top-to-bottom).
 * Board grid: col = 9 - file, row = rank.charCodeAt - 'a'.charCodeAt
 */
export function parseUsi(usi) { ... }
```

Returns `{fromIdx: number | null, toIdx: number, isDrop: boolean, dropPiece: string | null}` or `null` for unparseable input. Vitest covers board move, board promotion, drop, malformed input.

**`webui/src/lib/Board.svelte`:**

- Replace `export let lastMoveIdx = -1` with `export let lastMoveFromIdx = -1` and `export let lastMoveToIdx = -1`.
- Add `export let heatmap = null` (object `{squareIdx: probability}` or null).
- Replace `class:last-move={idx === lastMoveIdx}` with `class:last-move-to={idx === lastMoveToIdx}` and `class:last-move-from={idx === lastMoveFromIdx}`.
- Render heatmap as a per-square absolutely-positioned overlay div with `style="background: hsla(45, 90%, 55%, {prob * 0.5})"` when the square has an entry. Pointer-events disabled so it doesn't block anything.

**CSS:**

- `.last-move-to` keeps the existing `--bg-last-move` background and `--accent-teal` border.
- `.last-move-from` gets a thin gold inset border, no fill: `box-shadow: inset 0 0 0 2px var(--accent-gold)`.
- `.heatmap-overlay` fills the square at the chosen alpha, with `mix-blend-mode: multiply` so the kanji stays legible regardless of board theme.

**`webui/src/lib/ShowcaseView.svelte`:**

- Import `parseUsi` and a new `showcaseHeatmapEnabled` writable store.
- Compute `$: lastMoveCoords = move?.usi_notation ? parseUsi(move.usi_notation) : null`.
- Compute `$: heatmap = $showcaseHeatmapEnabled && move?.move_heatmap_json ? buildHeatmapMap(move.move_heatmap_json) : null` where `buildHeatmapMap` parses JSON `{usi: prob}` and converts each USI to a destination square idx via `parseUsi`. **Promotion variants** (e.g., `"5g5f"` and `"5g5f+"`) share the same destination square — `buildHeatmapMap` *sums* their probabilities into one entry, since from a spectator's perspective both represent "the model considered going to 5f".
- Add a small toggle button near the board header (alongside the existing player names): `<button on:click={() => showcaseHeatmapEnabled.update(v => !v)} aria-pressed={$showcaseHeatmapEnabled}>Heatmap</button>`.
- Pass new props to `<Board>`: `lastMoveFromIdx={lastMoveCoords?.fromIdx ?? -1} lastMoveToIdx={lastMoveCoords?.toIdx ?? -1} heatmap={heatmap}`.

**New store `webui/src/stores/showcase.js`:**

Add `showcaseHeatmapEnabled` writable backed by localStorage, mirroring the pattern used in `audio.js` / `theme.js`. Default `false`.

**`webui/src/App.svelte`:**

Remove the dead `lastMoveIdx` IIFE (`App.svelte:55-61`) and the `lastMoveIdx` prop on the training-tab Board (`App.svelte:204`). The Board prop signature now uses from/to indices, so the training-tab call site simply omits both — they default to -1 and render nothing. Out of scope to add real last-move tracking on the training tab.

### Visual treatment

| Element | Treatment |
|---------|-----------|
| Last-move from-square | Thin gold inset border (`var(--accent-gold)`), no fill |
| Last-move to-square | Existing `var(--bg-last-move)` fill + `var(--accent-teal)` border |
| Heatmap square | Warm overlay `hsla(45, 90%, 55%, prob × 0.5)`, `mix-blend-mode: multiply` |
| Heatmap toggle | Compact button in showcase header, `aria-pressed` for screen readers |

The from-highlight and the heatmap can co-occur visually (the from-square is part of the heatmap set, since it's a legal destination of zero-distance — actually no, the from-square is the *origin*, not a candidate destination, so they won't visually overlap unless the moved piece could have stayed put, which never happens in shogi).

## Storage cost

- Average legal moves from a single from-square: 5–17 (bishop range > knight).
- Per entry: ~25 bytes JSON (`"7g7f":0.4321,`).
- ~10 entries × ~25 bytes ≈ ~250 bytes per ply for board moves.
- Drops: 30–60 legal squares × ~20 bytes ≈ 1–1.2 KB per drop ply.
- ~150-ply game with ~20% drops ≈ ~70 KB total. Acceptable.

## Testing

**Rust (`shogi-gym/.venv` + maturin):**
- New unit test in `spectator.rs#[cfg(test)]`: `legal_moves_with_usi()` at startpos returns 30 entries, all USI strings parse via existing helpers, indices match `legal_actions()`.

**Python (`uv run pytest`):**
- Unit test for `_build_heatmap()` helper: board-move case (chosen `"7g7f"` filters to other moves starting `"7g"`), drop case (chosen `"P*5e"` filters to other `"P*"` moves), edge cases (chosen move's own USI included, NaN probabilities omitted).
- Existing `test_showcase_db.py` updated to verify the new column round-trips through `write_showcase_move()`.

**Vitest (`webui`):**
- `usiCoords.test.js`: board move, board move with promotion, drop, malformed input.
- `Board.svelte` rendering test (if existing pattern supports it): from/to highlight classes applied to correct indices; heatmap overlay rendered for entries in the map.
- `showcase.test.js`: `showcaseHeatmapEnabled` persists to localStorage.

**Manual verification:**
- Start the showcase sidecar with a real model.
- Confirm last-move from/to highlights appear and update each ply.
- Toggle heatmap on; confirm shading appears on candidate squares; confirm intensity correlates with how concentrated the policy was (a forced move shows a near-uniform single-square shade; a wide-open mid-game position shows multiple lighter shades).
- Toggle off; confirm overlay disappears.
- Reload page; confirm toggle state persisted.
- Confirm pre-existing match plies (NULL `move_heatmap_json`) show no overlay even with toggle on.

## Migration & deployment

- Schema bump v6 → v7 is purely additive; idempotent via `_migrate_add_column`.
- Maturin rebuild required: `cd shogi-engine/crates/shogi-gym && source .venv/bin/activate && maturin develop`. Document this in the implementation plan handoff.
- Frontend rebuild via existing vite tooling.
- No coordinated rollout needed; old data renders gracefully with the new code.

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Maturin rebuild step is forgotten in deployment | Implementation plan includes an explicit rebuild step; runtime failure (missing PyO3 method) would surface immediately at sidecar startup. |
| `parseUsi` becomes a duplicated source of truth for coordinate mapping | Co-locate vitest spec next to the helper; one place to grep for "usi" if conventions ever change. The Rust side's `square_notation` is the canonical Hodges definition — comment in `parseUsi` references it. |
| Heatmap overlay obscures kanji on dark themes | `mix-blend-mode: multiply` keeps the piece glyph dominant; manual verification on both light and dark themes covers this. |
| Player-perspective flipping (white-side view) breaks coordinate math | Board.svelte does not currently flip for white perspective — squares are always rendered from black's POV. `parseUsi` matches that convention. If perspective-flipping is added later, both `parseUsi` consumers and the board rendering would need to be updated together. Out of scope here. |
