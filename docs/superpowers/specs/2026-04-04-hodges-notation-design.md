# Hodges Notation for Move Log

**Date:** 2026-04-04
**Status:** Draft
**Goal:** Replace coordinate notation (`9g→9f`) with standard Western (Hodges) shogi notation (`P-7f`) so that move logs are readable by frontier LLMs for gameplay review.

## Notation Format

Follows the Hodges system (1976) — the most widely used Western shogi notation.

### Piece Letters

| Piece | Letter | Promoted |
|-------|--------|----------|
| King | K | — |
| Rook | R | +R |
| Bishop | B | +B |
| Gold | G | — |
| Silver | S | +S |
| Knight | N | +N |
| Lance | L | +L |
| Pawn | P | +P |

### Coordinates

File (column) 1–9, rank (row) a–i. File 1 is rightmost from Black's perspective, rank `a` is the top (White's back rank).

Internal mapping: `file = 9 - col`, `rank = (b'a' + row) as char`.

### Move Formats

| Type | Format | Example |
|------|--------|---------|
| Simple move | `{piece}-{dest}` | `P-7f` |
| Capture | `{piece}x{dest}` | `Bx3c` |
| Promotion | `{piece}{sep}{dest}+` | `Nx7c+` |
| Declined promotion | `{piece}{sep}{dest}=` | `S-4d=` |
| Promoted piece | `+{piece}{sep}{dest}` | `+R-5a` |
| Drop | `{piece}*{dest}` | `P*5e` |
| Disambiguated | `{piece}{origin}{sep}{dest}` | `G6g-7h` |

Where `{sep}` is `-` (move) or `x` (capture).

### Promotion Rules

Append `+` when `promote == true`.

Append `=` when ALL of:
- Board move (not drop)
- Piece base type `can_promote()` (P/L/N/S/B/R)
- Piece is not already promoted
- Move touches the promotion zone (`from` or `to` in zone)
- `promote == false`
- Promotion is NOT forced (see below)

Promotion zone: rows 0–2 (ranks a–c) for Black, rows 6–8 (ranks g–i) for White.

### Forced Promotion

Certain moves require mandatory promotion — the `=` suffix must never appear:
- **Pawn/Lance** reaching the last rank (rank a for Black, rank i for White)
- **Knight** reaching the last two ranks (ranks a-b for Black, ranks h-i for White)

The engine should never generate `promote == false` for these moves, but the notation function must guard against it: if promotion is forced, always emit `+` regardless of the `promote` flag. This makes notation correctness independent of engine invariants.

### Disambiguation

Include origin coordinates when another legal Board move exists with the same piece type, same promoted status, and same destination square.

Skip disambiguation for Kings (only one per side — never ambiguous).

## Implementation

### Rust: `spectator_data.rs`

Change signature:

```rust
pub fn move_notation(mv: Move, position: &Position, legal_moves: &[Move]) -> String
```

New helper functions:

```rust
fn piece_char(pt: PieceType) -> char
    // K, R, B, G, S, N, L, P

fn square_notation(sq: Square) -> String
    // e.g., "7f"

fn in_promotion_zone(sq: Square, color: Color) -> bool

fn could_promote(piece: Piece, from: Square, to: Square) -> bool

fn needs_disambiguation(mv: Move, position: &Position, legal_moves: &[Move]) -> bool
```

Logic for Board moves:
1. Look up piece at `from` → get type, color, promoted status
2. Check if `to` is occupied → capture (`x`) or move (`-`)
3. Build piece prefix: `+R` if promoted, `R` if not
4. Check disambiguation against `legal_moves` (skip for Kings)
5. Determine promotion suffix: `+`, `=`, or empty
6. Assemble: `{prefix}{origin?}{sep}{dest}{suffix}`

Logic for Drop moves (unchanged structure):
- `{piece_char}*{dest}` — e.g., `P*5e`

### Rust: Callers

**`vec_env.rs`** (line 679): Pass `&self.games[i].position` and legal moves. The legal move list from `write_obs_and_mask` is not cached (only the bitmask is kept), so we generate legal moves per-env at notation time using a shared `MoveList`. This is cheap — the same generation already runs every step for the mask — and only executes in the single-threaded notation recording loop, not the parallel apply phase.

**Borrow scoping note:** `generate_legal_moves_into` takes `&mut GameState`. The mutable borrow must end before passing `&game.position` and `&move_list` to `move_notation`. Scope this by generating legal moves into a local `MoveList` first, then calling `move_notation` with shared borrows only.

**`spectator.rs`**: Pass position from the SpectatorEnv's game state.

### Rust: `hand_piece_char` consolidation

The existing `hand_piece_char(HandPieceType)` maps the same letters as the new `piece_char(PieceType)`. Consolidate so both use one mapping. `hand_piece_char` can delegate to `piece_char` via `hpt.to_piece_type()`.

### WebUI: `moveRows.js`

The `toJapanese()` regex `/([1-9])([a-i])/g` still correctly matches coordinates in the new format. No changes needed for Japanese conversion.

The toggle button labels could update from `'coord'`/`'japanese'` to `'western'`/`'japanese'` for accuracy, but this is cosmetic.

### Tests

**Rust unit tests:**
- `move_notation()`: simple move, capture, promotion, declined promotion, promoted piece moving, drop, disambiguation (two same-type pieces to same dest), no disambiguation needed (single piece)
- `square_notation()`: corner squares `1a`, `9i`, `5e` to catch off-by-one in the `9 - col` / `a + row` mapping
- `in_promotion_zone()`: boundary rows — row 2 vs 3 for Black, row 5 vs 6 for White (fencepost errors)
- `could_promote()`: all 5 conditions tested independently — already promoted, non-promotable type (Gold/King), neither square in zone, drop move, promote flag true
- Forced promotion: pawn/lance on last rank always gets `+` never `=`; knight on last two ranks likewise
- King moves: never disambiguated
- `hand_piece_char` consolidation: regression test that all 7 hand piece types still map correctly

**WebUI tests:**
- Update `moveRows.test.js` fixtures to use new Hodges notation format
- Add integration smoke test: Hodges notation strings (simple, promoted piece, drop) through `toJapanese()` produce correct output

**Existing test updates:**
- Spectator tests that assert on notation strings need updating to match new format

## Scope Exclusions

- No migration of old DB data (no history to preserve)
- No changes to action encoding or training pipeline
- No changes to the `toJapanese()` converter logic (just its input strings change)

## Review Notes

Spec reviewed by 5 specialist agents (architecture, systems, Rust, QA, shogi). Key changes incorporated:

1. **`legal_moves` made mandatory** (`&[Move]` not `Option`) — silent disambiguation failure is a correctness bug for the LLM use case (Architecture, Rust, Shogi)
2. **Disambiguation example fixed** — `G67-78` → `G6g-7h` (Shogi)
3. **Forced promotion guard added** — P/L on last rank, N on last two ranks must always promote (Shogi)
4. **King disambiguation skip** — only one per side, never ambiguous (Shogi)
5. **Borrow scoping documented** — `&mut` for movegen must end before `&` borrows for notation (Rust)
6. **Test plan expanded** — boundary tests, 5-condition `=` guard, square_notation corners, integration smoke test (QA)

Deferred to future work: structured tuple storage for multi-format rendering (Systems), verbose mode with always-on origin (Shogi)
