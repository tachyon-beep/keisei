# Showcase Last-Move Indicator + Policy Heatmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add always-on last-move from/to highlighting and a toggleable policy-preference heatmap overlay to the showcase board.

**Architecture:** New PyO3 method `SpectatorEnv.legal_moves_with_usi()` exposes legal moves with USI strings → runner builds a `{usi: probability}` dict filtered to the chosen move's from-square (or drop prefix) → stored in a new `showcase_moves.move_heatmap_json` column (additive v6→v7 migration) → flows through existing WS payload → Svelte parses USI to board indices and renders highlights + alpha-scaled overlay.

**Tech Stack:** Rust (PyO3 / shogi-gym), Python 3.13 (FastAPI / SQLite), Svelte 4 (Vite + Vitest).

**Spec:** `docs/superpowers/specs/2026-05-03-showcase-last-move-heatmap-design.md`

---

## File Structure

**Created:**
- `keisei/showcase/heatmap.py` — pure helper `build_heatmap()` (testable without env)
- `tests/test_showcase_heatmap.py` — unit tests for the helper
- `webui/src/lib/usiCoords.js` — `parseUsi()` helper
- `webui/src/lib/usiCoords.test.js` — vitest

**Modified:**
- `shogi-engine/crates/shogi-gym/src/spectator.rs` — add `legal_moves_with_usi()` + Rust unit test
- `keisei/db.py` — bump `SCHEMA_VERSION` to 7, add column to `showcase_moves` DDL, add `_migrate_v6_to_v7`, register migration
- `keisei/showcase/db_ops.py` — extend `write_showcase_move()` signature + INSERT
- `keisei/showcase/runner.py` — call helper, pass result to write
- `tests/test_showcase_db.py` — round-trip the new column
- `webui/src/lib/Board.svelte` — replace `lastMoveIdx` with `lastMoveFromIdx`+`lastMoveToIdx`, add `heatmap` prop + overlay rendering
- `webui/src/lib/ShowcaseView.svelte` — toggle button, USI parsing, prop wiring
- `webui/src/stores/showcase.js` — add `showcaseHeatmapEnabled` writable
- `webui/src/stores/showcase.test.js` — cover the new store
- `webui/src/App.svelte` — drop dead `lastMoveIdx` IIFE and prop on training-tab Board

**Untouched (verified):** `keisei/server/app.py` and `webui/src/lib/ws.js` use `dict(row)` / `msg.new_moves` patterns that pass the new column through automatically.

---

## Task 1: Rust — `legal_moves_with_usi()` PyO3 method

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/spectator.rs` (add method near `legal_actions` at line 206; add test in `#[cfg(test)] mod tests`)

- [ ] **Step 1: Add the failing Rust unit test**

Append to the `mod tests` block at the bottom of `spectator.rs`:

```rust
#[test]
fn test_legal_moves_with_usi_startpos_matches_legal_actions() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|_py| {
        let mut env = SpectatorEnv::new(Some(512), Some("spatial".to_string())).unwrap();
        env.reset_internal();
        let actions = env.legal_actions();
        let with_usi = env.legal_moves_with_usi();
        assert_eq!(actions.len(), with_usi.len(), "counts must match");
        assert_eq!(actions.len(), 30, "startpos has 30 legal moves");
        let action_set: std::collections::HashSet<usize> = actions.into_iter().collect();
        for (idx, usi) in &with_usi {
            assert!(action_set.contains(idx), "action {} from with_usi missing from legal_actions", idx);
            assert!(!usi.is_empty(), "USI must not be empty");
            // Startpos has only board moves; no drops
            assert!(!usi.contains('*'), "no drops at startpos: {}", usi);
            assert_eq!(usi.len(), 4, "startpos USI is 4 chars (no promotion possible): {}", usi);
        }
    });
}
```

If `SpectatorEnv::new` or `reset_internal` have different signatures, mirror the pattern used in the nearest existing test (around line 337). The exact call surface is less important than: construct env, reset, compare counts and contents.

- [ ] **Step 2: Run the test to verify it fails to compile**

```bash
cd shogi-engine
cargo test -p shogi-gym test_legal_moves_with_usi 2>&1 | head -40
```

Expected: compile error on `env.legal_moves_with_usi()` — method not found. (Note: `cargo test -p shogi-gym` cannot fully run because shogi-gym is a cdylib needing Python symbols — but the *compile* failure is the meaningful signal here. Final test execution happens via `pytest` after maturin develop in step 5.)

If the cargo build fails entirely with link errors before the missing-method error, accept that; the real test run is in step 5.

- [ ] **Step 3: Implement the method**

Add this method to the `#[pymethods] impl SpectatorEnv` block, right after `legal_actions` (~line 215):

```rust
/// Return all legal moves at the current position with their USI strings.
/// Read-only; no state mutation. Order matches `legal_actions()`.
pub fn legal_moves_with_usi(&mut self) -> Vec<(usize, String)> {
    use crate::spectator_data::move_usi;
    let perspective = self.game.position.current_player;
    let moves = self.game.legal_moves();
    moves
        .into_iter()
        .map(|mv| {
            let idx = self.mapper.encode(mv, perspective)
                .expect("legal move must be encodable");
            (idx, move_usi(mv))
        })
        .collect()
}
```

If `move_usi` is already in scope at the top of `spectator.rs`, drop the `use` line.

- [ ] **Step 4: Build via maturin and run via pytest**

```bash
cd shogi-engine/crates/shogi-gym
source .venv/bin/activate
maturin develop 2>&1 | tail -20
```

Expected: clean build. Then verify the method is exposed in Python:

```bash
python -c "from shogi_gym import SpectatorEnv; e = SpectatorEnv(max_ply=512, action_mode='spatial'); e.reset(); m = e.legal_moves_with_usi(); print(len(m), m[:3])"
```

Expected: `30` followed by three `(int, "XYxy")` tuples.

- [ ] **Step 5: Run the Rust unit test through the cargo path that links Python**

The shogi-gym tests run via the python harness when the maturin extension is loaded. From the same activated venv:

```bash
cd /home/john/keisei/shogi-engine/crates/shogi-gym
python -m pytest tests/ -k "legal" -v 2>&1 | tail -20
```

Expected: existing legal-action tests still pass. (Pure Rust `cargo test -p shogi-gym` cannot run for cdylib reasons — see project CLAUDE.md.)

If a Rust unit test really needs to live in `#[cfg(test)]`, accept that it will be exercised when shogi-gym is rebuilt as part of normal development; the `python -c` smoke check above is the binding contract.

- [ ] **Step 6: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/spectator.rs
git commit -m "feat(shogi-gym): expose legal_moves_with_usi() on SpectatorEnv"
```

---

## Task 2: Python — pure heatmap helper

**Files:**
- Create: `keisei/showcase/heatmap.py`
- Create: `tests/test_showcase_heatmap.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_showcase_heatmap.py`:

```python
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


def test_missing_action_index_in_probs_is_skipped() -> None:
    """Defensive: legal moves whose index isn't in probs are silently skipped."""
    legal = [(10, "7g7f"), (99, "7g7e")]
    probs = {10: 0.95}  # 99 missing
    out = build_heatmap(chosen_usi="7g7f", legal_with_usi=legal, probs=probs)
    assert out == {"7g7f": pytest.approx(0.95)}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_showcase_heatmap.py -v
```

Expected: ImportError — module doesn't exist yet.

- [ ] **Step 3: Implement the helper**

Create `keisei/showcase/heatmap.py`:

```python
"""Build the policy-preference heatmap for a showcase ply.

The heatmap is a {usi: probability} dict containing legal moves that share
the chosen move's from-square (board moves) or piece type (drops). It is
serialized to JSON and stored in showcase_moves.move_heatmap_json.
"""
from __future__ import annotations

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
        if prob is None or prob <= 0.0:
            continue
        out[usi] = float(prob)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_showcase_heatmap.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add keisei/showcase/heatmap.py tests/test_showcase_heatmap.py
git commit -m "feat(showcase): add build_heatmap() helper for policy-preference overlay"
```

---

## Task 3: SQLite — schema migration v6 → v7

**Files:**
- Modify: `keisei/db.py` (`SCHEMA_VERSION` constant ~line 13; `showcase_moves` DDL ~line 420; new migration after `_migrate_v5_to_v6` ~line 132; `MIGRATIONS` dict ~line 151)

- [ ] **Step 1: Bump SCHEMA_VERSION**

In `keisei/db.py`, change line 13:

```python
SCHEMA_VERSION = 7
```

- [ ] **Step 2: Add the column to the showcase_moves DDL**

In the `CREATE TABLE IF NOT EXISTS showcase_moves` block (~line 420), add `move_heatmap_json TEXT` between `top_candidates TEXT` and `move_time_ms INTEGER`:

```sql
CREATE TABLE IF NOT EXISTS showcase_moves (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         INTEGER NOT NULL REFERENCES showcase_games(id),
    ply             INTEGER NOT NULL,
    action_index    INTEGER NOT NULL,
    usi_notation    TEXT NOT NULL,
    board_json      TEXT NOT NULL,
    hands_json      TEXT NOT NULL,
    current_player  TEXT NOT NULL,
    in_check        INTEGER NOT NULL DEFAULT 0,
    value_estimate  REAL,
    top_candidates  TEXT,
    move_heatmap_json TEXT,
    move_time_ms    INTEGER,
    created_at      TEXT NOT NULL,
    UNIQUE(game_id, ply)
);
```

- [ ] **Step 3: Add the migration function**

After `_migrate_v5_to_v6()` (~line 149), insert:

```python
def _migrate_v6_to_v7(conn: sqlite3.Connection) -> None:
    """v6 -> v7: Add showcase_moves.move_heatmap_json column.

    Stores a JSON {usi: probability} dict containing legal moves sharing the
    chosen move's from-square (or drop prefix), used by the showcase tab's
    toggleable policy-preference heatmap overlay. Nullable — pre-migration
    rows render no heatmap, which is the correct fallback.
    """
    _migrate_add_column(conn, "showcase_moves", "move_heatmap_json", "TEXT")
```

- [ ] **Step 4: Register the migration**

In the `MIGRATIONS` dict (~line 151), add the new entry:

```python
MIGRATIONS = {
    2: _migrate_v1_to_v2,
    3: _migrate_v2_to_v3,
    4: _migrate_v3_to_v4,
    5: _migrate_v4_to_v5,
    6: _migrate_v5_to_v6,
    7: _migrate_v6_to_v7,
}
```

- [ ] **Step 5: Verify the migration runs cleanly on a fresh DB**

```bash
uv run python -c "
import tempfile, os, sqlite3
from keisei.db import init_db, _connect
with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    path = f.name
try:
    init_db(path)
    conn = _connect(path)
    cols = [r[1] for r in conn.execute('PRAGMA table_info(showcase_moves)').fetchall()]
    print('columns:', cols)
    assert 'move_heatmap_json' in cols, 'column missing'
    print('OK')
finally:
    os.unlink(path)
"
```

Expected: column list includes `move_heatmap_json`, then `OK`.

- [ ] **Step 6: Commit**

```bash
git add keisei/db.py
git commit -m "feat(db): v6->v7 migration adds showcase_moves.move_heatmap_json"
```

---

## Task 4: Extend `write_showcase_move()` to accept the new column

**Files:**
- Modify: `keisei/showcase/db_ops.py:124-153`
- Modify: `tests/test_showcase_db.py` (extend an existing test or add one)

- [ ] **Step 1: Write the failing round-trip test**

Add to `tests/test_showcase_db.py` inside the `TestGameOperations` class:

```python
    def test_write_and_read_move_heatmap_json(self, db: str) -> None:
        """move_heatmap_json round-trips through write_showcase_move."""
        qid = queue_match(db, "e1", "e2", "normal")
        game_id = create_showcase_game(
            db, queue_id=qid, entry_id_black="e1", entry_id_white="e2",
            elo_black=1500.0, elo_white=1480.0, name_black="A", name_white="B",
        )
        write_showcase_move(
            db, game_id=game_id, ply=1, action_index=42,
            usi_notation="7g7f", board_json="[]", hands_json="{}",
            current_player="white", in_check=False, value_estimate=0.52,
            top_candidates='[]', move_time_ms=15,
            move_heatmap_json='{"7g7f": 0.5, "7g7e": 0.3}',
        )
        moves = read_showcase_moves_since(db, game_id, since_ply=0)
        assert moves[0]["move_heatmap_json"] == '{"7g7f": 0.5, "7g7e": 0.3}'

    def test_write_move_heatmap_defaults_to_none(self, db: str) -> None:
        """Omitting move_heatmap_json leaves the column NULL (backward compat)."""
        qid = queue_match(db, "e1", "e2", "normal")
        game_id = create_showcase_game(
            db, queue_id=qid, entry_id_black="e1", entry_id_white="e2",
            elo_black=1500.0, elo_white=1480.0, name_black="A", name_white="B",
        )
        write_showcase_move(
            db, game_id=game_id, ply=1, action_index=42,
            usi_notation="7g7f", board_json="[]", hands_json="{}",
            current_player="white", in_check=False, value_estimate=0.52,
            top_candidates='[]', move_time_ms=15,
        )
        moves = read_showcase_moves_since(db, game_id, since_ply=0)
        assert moves[0]["move_heatmap_json"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_showcase_db.py::TestGameOperations::test_write_and_read_move_heatmap_json tests/test_showcase_db.py::TestGameOperations::test_write_move_heatmap_defaults_to_none -v
```

Expected: TypeError / KeyError on `move_heatmap_json` parameter.

- [ ] **Step 3: Update `write_showcase_move()` signature and INSERT**

Replace the function in `keisei/showcase/db_ops.py:124-153`:

```python
def write_showcase_move(db_path: str, *, game_id: int, ply: int, action_index: int,
                         usi_notation: str, board_json: str, hands_json: str,
                         current_player: str, in_check: bool, value_estimate: float,
                         top_candidates: str, move_time_ms: int,
                         move_heatmap_json: str | None = None) -> None:
    """Atomic write: INSERT move + UPDATE total_ply in one transaction."""
    conn = _connect(db_path)
    try:
        now = _now_iso()
        for attempt in range(MAX_RETRIES):
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """INSERT OR IGNORE INTO showcase_moves
                       (game_id, ply, action_index, usi_notation, board_json, hands_json,
                        current_player, in_check, value_estimate, top_candidates,
                        move_heatmap_json, move_time_ms, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (game_id, ply, action_index, usi_notation, board_json, hands_json,
                     current_player, int(in_check), value_estimate, top_candidates,
                     move_heatmap_json, move_time_ms, now))
                conn.execute("UPDATE showcase_games SET total_ply = ? WHERE id = ?", (ply, game_id))
                conn.commit()
                return
            except sqlite3.OperationalError as e:
                conn.rollback()
                if "database is locked" in str(e) and attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.05)
                    time.sleep(delay)
                else:
                    raise
    finally:
        conn.close()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_showcase_db.py -v 2>&1 | tail -20
```

Expected: all `test_showcase_db` tests pass, including the two new ones. Existing tests that don't pass `move_heatmap_json` still work because of the default.

- [ ] **Step 5: Commit**

```bash
git add keisei/showcase/db_ops.py tests/test_showcase_db.py
git commit -m "feat(showcase): write_showcase_move() accepts move_heatmap_json"
```

---

## Task 5: Wire the heatmap into `runner.py`

**Files:**
- Modify: `keisei/showcase/runner.py:140-192`

- [ ] **Step 1: Update the runner to call the helper and pass it through**

Modify the move-emission block in `_run_game()`. Current code (~lines 141-192) needs three additions:

1. Import `build_heatmap` near the top (after the other `keisei.showcase` imports ~line 38):

```python
from keisei.showcase.heatmap import build_heatmap
```

2. After computing `probs` (~line 163) and *before* `env.step(action)` (~line 175), capture the legal-with-USI list and the chosen USI's heatmap. The chosen action's USI isn't known until *after* `env.step()` populates `move_history`, so we have to delay heatmap construction until then.

Restructure the block so `legal_moves_with_usi` is captured BEFORE the step (the position must match the policy distribution), and the heatmap is built AFTER the chosen USI is known:

Replace lines 146-192 with:

```python
                legal = env.legal_actions()
                # Capture USIs for the heatmap before stepping — position must
                # match the policy distribution we just computed.
                legal_with_usi = env.legal_moves_with_usi()
                mask = np.full(policy_logits.shape, -1e9)
                mask[legal] = 0.0
                masked_logits = policy_logits + mask

                # Temperature-scaled softmax over legal moves only (S3: NaN guard)
                scaled_logits = masked_logits / SAMPLING_TEMPERATURE
                legal_logits = scaled_logits[legal]
                legal_probs = np.exp(legal_logits - legal_logits.max())
                total = legal_probs.sum()
                if total < 1e-10:
                    legal_probs = np.ones(len(legal)) / len(legal)
                else:
                    legal_probs = legal_probs / total

                # Full probability array for top-candidates display
                probs = np.zeros_like(scaled_logits)
                probs[legal] = legal_probs

                top_indices = np.argsort(probs)[::-1][:3]
                top_candidates = []
                for idx in top_indices:
                    if probs[idx] > 0.001:
                        top_candidates.append({"action": int(idx), "probability": round(float(probs[idx]), 4)})

                # Sample from legal moves only (avoids illegal-action residual risk)
                chosen_idx = int(np.random.choice(len(legal), p=legal_probs))
                action = legal[chosen_idx]

                state = env.step(action)
                ply = state["ply"]

                if state["move_history"]:
                    usi_notation = state["move_history"][-1]["notation"]
                else:
                    usi_notation = f"action_{action}"

                for tc in top_candidates:
                    tc["usi"] = usi_notation if tc["action"] == action else f"a{tc['action']}"

                # Build heatmap of moves sharing the chosen move's from-square
                # (board move) or drop prefix (drop). Lean: only same-prefix.
                heatmap = build_heatmap(
                    chosen_usi=usi_notation,
                    legal_with_usi=legal_with_usi,
                    probs={int(i): float(probs[i]) for i in legal},
                )

                write_showcase_move(
                    self.db_path, game_id=game_id, ply=ply, action_index=action,
                    usi_notation=usi_notation, board_json=json.dumps(state["board"]),
                    hands_json=json.dumps(state["hands"]), current_player=state["current_player"],
                    in_check=state.get("in_check", False), value_estimate=win_prob,
                    top_candidates=json.dumps(top_candidates), move_time_ms=inference_ms,
                    move_heatmap_json=json.dumps(heatmap),
                )
```

The only changes vs. existing code: added `legal_with_usi = env.legal_moves_with_usi()` line, added `heatmap = build_heatmap(...)` block, added `move_heatmap_json=json.dumps(heatmap)` kwarg.

- [ ] **Step 2: Run all showcase tests + a smoke check**

```bash
uv run pytest tests/test_showcase_db.py tests/test_showcase_heatmap.py -v 2>&1 | tail -10
```

Expected: all pass.

Smoke check that runner.py imports cleanly:

```bash
uv run python -c "from keisei.showcase.runner import ShowcaseRunner; print('OK')"
```

Expected: `OK`. (No need to run a full game in tests — that's manual verification later.)

- [ ] **Step 3: Commit**

```bash
git add keisei/showcase/runner.py
git commit -m "feat(showcase): emit move_heatmap_json per ply in runner"
```

---

## Task 6: Frontend — `parseUsi` helper

**Files:**
- Create: `webui/src/lib/usiCoords.js`
- Create: `webui/src/lib/usiCoords.test.js`

- [ ] **Step 1: Write the failing tests**

Create `webui/src/lib/usiCoords.test.js`:

```javascript
import { describe, it, expect } from 'vitest'
import { parseUsi } from './usiCoords.js'

describe('parseUsi', () => {
  // Coordinate convention check:
  //   "9a" -> file=9, rank=a -> col = 9-9 = 0, row = 0 -> idx 0 (top-left)
  //   "1i" -> file=1, rank=i -> col = 9-1 = 8, row = 8 -> idx 80 (bottom-right)
  //   "5e" -> file=5, rank=e -> col = 9-5 = 4, row = 4 -> idx 40 (centre)

  it('parses a simple board move', () => {
    // 7g -> col=2, row=6 -> idx 6*9+2 = 56
    // 7f -> col=2, row=5 -> idx 5*9+2 = 47
    expect(parseUsi('7g7f')).toEqual({
      fromIdx: 56, toIdx: 47, isDrop: false, dropPiece: null,
    })
  })

  it('parses a board move with promotion suffix', () => {
    expect(parseUsi('8h2b+')).toEqual({
      fromIdx: 64, toIdx: 16, isDrop: false, dropPiece: null,
    })
  })

  it('parses a drop', () => {
    // P*5e -> drop pawn at 5e -> idx 40
    expect(parseUsi('P*5e')).toEqual({
      fromIdx: null, toIdx: 40, isDrop: true, dropPiece: 'P',
    })
  })

  it('parses corner squares', () => {
    expect(parseUsi('9a1i')).toEqual({
      fromIdx: 0, toIdx: 80, isDrop: false, dropPiece: null,
    })
  })

  it('returns null for malformed input', () => {
    expect(parseUsi('')).toBeNull()
    expect(parseUsi('garbage')).toBeNull()
    expect(parseUsi('zz')).toBeNull()
    expect(parseUsi(null)).toBeNull()
    expect(parseUsi(undefined)).toBeNull()
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd webui && npx vitest run src/lib/usiCoords.test.js 2>&1 | tail -15
```

Expected: failure on import — module doesn't exist.

- [ ] **Step 3: Implement the helper**

Create `webui/src/lib/usiCoords.js`:

```javascript
/**
 * Parse a USI move string and return board indices (0-80, row*9+col).
 *
 * USI conventions (matches Rust shogi-gym/src/spectator_data.rs#square_notation):
 *   File: digit '1'-'9', counted right-to-left from black's POV. col = 9 - file.
 *   Rank: letter 'a'-'i', top-to-bottom. row = rank.charCodeAt - 'a'.charCodeAt.
 *   Board move: "<from-file><from-rank><to-file><to-rank>[+]"  e.g. "7g7f", "8h2b+"
 *   Drop:       "<piece-char>*<to-file><to-rank>"               e.g. "P*5e"
 *
 * @param {string} usi
 * @returns {{fromIdx: number|null, toIdx: number, isDrop: boolean, dropPiece: string|null}|null}
 */
export function parseUsi(usi) {
  if (typeof usi !== 'string' || usi.length < 2) return null

  // Drop: "P*5e"
  if (usi[1] === '*') {
    if (usi.length < 4) return null
    const piece = usi[0]
    const toIdx = squareToIdx(usi.slice(2, 4))
    if (toIdx === null) return null
    return { fromIdx: null, toIdx, isDrop: true, dropPiece: piece }
  }

  // Board move: "7g7f" or "7g7f+"
  if (usi.length < 4) return null
  const fromIdx = squareToIdx(usi.slice(0, 2))
  const toIdx = squareToIdx(usi.slice(2, 4))
  if (fromIdx === null || toIdx === null) return null
  return { fromIdx, toIdx, isDrop: false, dropPiece: null }
}

function squareToIdx(sq) {
  if (sq.length !== 2) return null
  const file = sq.charCodeAt(0) - '1'.charCodeAt(0) + 1   // '1'..'9' -> 1..9
  const rank = sq.charCodeAt(1) - 'a'.charCodeAt(0)        // 'a'..'i' -> 0..8
  if (file < 1 || file > 9 || rank < 0 || rank > 8) return null
  const col = 9 - file
  return rank * 9 + col
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd webui && npx vitest run src/lib/usiCoords.test.js
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add webui/src/lib/usiCoords.js webui/src/lib/usiCoords.test.js
git commit -m "feat(webui): parseUsi helper for board-index extraction"
```

---

## Task 7: Frontend — `showcaseHeatmapEnabled` store

**Files:**
- Modify: `webui/src/stores/showcase.js` (append new store, mirroring `audio.js` pattern)
- Modify: `webui/src/stores/showcase.test.js`

- [ ] **Step 1: Write the failing test**

Append to `webui/src/stores/showcase.test.js`:

```javascript
import { showcaseHeatmapEnabled } from './showcase.js'
import { get } from 'svelte/store'

describe('showcaseHeatmapEnabled', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('defaults to false when localStorage is empty', () => {
    // Re-import would be cleaner, but the module is already evaluated.
    // The default-from-empty assertion is covered by initial-state on a fresh DOM.
    expect(typeof get(showcaseHeatmapEnabled)).toBe('boolean')
  })

  it('persists value to localStorage when toggled', () => {
    showcaseHeatmapEnabled.set(true)
    expect(localStorage.getItem('showcaseHeatmapEnabled')).toBe('true')
    showcaseHeatmapEnabled.set(false)
    expect(localStorage.getItem('showcaseHeatmapEnabled')).toBe('false')
  })
})
```

(The "defaults to false on fresh load" assertion is best left to a one-off manual check or a separate test file with module-reload — vitest module caching makes it awkward to assert default cleanly mid-suite.)

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd webui && npx vitest run src/stores/showcase.test.js 2>&1 | tail -15
```

Expected: failure on import — `showcaseHeatmapEnabled` not exported.

- [ ] **Step 3: Add the store**

Append to `webui/src/stores/showcase.js`:

```javascript
/** Persisted toggle for the showcase board's policy heatmap overlay. */
const HEATMAP_KEY = 'showcaseHeatmapEnabled'

function loadHeatmapInitial() {
  if (typeof localStorage === 'undefined') return false
  return localStorage.getItem(HEATMAP_KEY) === 'true'
}

export const showcaseHeatmapEnabled = writable(loadHeatmapInitial())

showcaseHeatmapEnabled.subscribe((val) => {
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(HEATMAP_KEY, val ? 'true' : 'false')
  }
})
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd webui && npx vitest run src/stores/showcase.test.js
```

Expected: all showcase store tests pass.

- [ ] **Step 5: Commit**

```bash
git add webui/src/stores/showcase.js webui/src/stores/showcase.test.js
git commit -m "feat(webui): add localStorage-backed showcaseHeatmapEnabled store"
```

---

## Task 8: Frontend — `Board.svelte` prop changes + heatmap rendering

**Files:**
- Modify: `webui/src/lib/Board.svelte`

- [ ] **Step 1: Update the props and rendering**

Replace the `<script>` block (lines 1-21) with:

```svelte
<script>
  import { pieceKanji } from './pieces.js'

  export let board = []
  export let inCheck = false
  export let currentPlayer = 'black'
  /** Index (0-80) of the last move's from-square, or -1 (drops, or no move yet). */
  export let lastMoveFromIdx = -1
  /** Index (0-80) of the last move's destination square, or -1. */
  export let lastMoveToIdx = -1
  /**
   * Optional policy heatmap: object mapping board index (0-80) to probability (0-1).
   * Pass null or omit to render no overlay.
   */
  export let heatmap = null

  const colLabels = [9, 8, 7, 6, 5, 4, 3, 2, 1]
  const rowLabels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

  // Summarize piece positions for screen readers (view-only board)
  $: boardDescription = (() => {
    const counts = { black: 0, white: 0 }
    for (const piece of board) {
      if (piece) counts[piece.color]++
    }
    return `Black has ${counts.black} pieces, White has ${counts.white} pieces on the board`
  })()
</script>
```

Replace the square render block (lines 32-50) with:

```svelte
      {#each Array(81) as _, idx}
        {@const piece = board[idx]}
        {@const heatProb = heatmap?.[idx] ?? null}
        <div
          class="square"
          class:has-piece={piece != null}
          class:last-move-to={idx === lastMoveToIdx}
          class:last-move-from={idx === lastMoveFromIdx}
        >
          {#if heatProb !== null && heatProb > 0}
            <div class="heatmap-overlay" style="opacity: {Math.min(1, heatProb * 0.5 + 0.15)};" aria-hidden="true"></div>
          {/if}
          {#if piece}
            <span
              class="piece"
              class:white={piece.color === 'white'}
              class:promoted={piece.promoted}
              lang="ja"
            >
              {pieceKanji(piece.type, piece.promoted, piece.color)}
            </span>
          {/if}
        </div>
      {/each}
```

Replace the `.square.last-move` CSS rule (lines 105-108) with:

```css
  .square.last-move-to {
    background: var(--bg-last-move);
    border-color: var(--accent-teal);
  }

  .square.last-move-from {
    box-shadow: inset 0 0 0 2px var(--accent-gold);
  }

  .heatmap-overlay {
    position: absolute;
    inset: 0;
    background: hsl(45, 90%, 55%);
    mix-blend-mode: multiply;
    pointer-events: none;
  }
```

The `+ 0.15` floor in the opacity expression keeps very low-probability moves still faintly visible (otherwise tiny probabilities would render as transparent and be indistinguishable from non-candidate squares).

- [ ] **Step 2: Verify the build still compiles**

```bash
cd webui && npx vite build 2>&1 | tail -10
```

Expected: build succeeds. Vite/Svelte will surface any syntax issues.

- [ ] **Step 3: Commit**

```bash
git add webui/src/lib/Board.svelte
git commit -m "feat(webui): Board accepts last-move from/to + heatmap overlay"
```

---

## Task 9: Frontend — `ShowcaseView.svelte` toggle + wiring

**Files:**
- Modify: `webui/src/lib/ShowcaseView.svelte`

- [ ] **Step 1: Wire up the parser, store, and toggle**

Replace the `<script>` block (lines 1-23) with:

```svelte
<script>
  import { showcaseGame, showcaseMoves, showcaseCurrentMove, sidecarAlive, showcaseHeatmapEnabled } from '../stores/showcase.js'
  import { safeParse } from './safeParse.js'
  import { parseUsi } from './usiCoords.js'
  import Board from './Board.svelte'
  import PieceTray from './PieceTray.svelte'
  import MoveLog from './MoveLog.svelte'
  import EvalBar from './EvalBar.svelte'
  import MatchControls from './MatchControls.svelte'
  import CommentaryPanel from './CommentaryPanel.svelte'
  import WinProbGraph from './WinProbGraph.svelte'
  import MatchQueue from './MatchQueue.svelte'

  $: move = $showcaseCurrentMove
  $: board = move ? safeParse(move.board_json, []) : []
  $: hands = move ? safeParse(move.hands_json, {}) : {}
  $: game = $showcaseGame
  $: moveHistoryJson = JSON.stringify(
    ($showcaseMoves || []).map(m => ({
      action: m.action_index,
      notation: m.usi_notation,
    }))
  )

  // Last-move highlight: parse the current move's USI for from/to indices.
  $: lastMoveCoords = move?.usi_notation ? parseUsi(move.usi_notation) : null
  $: lastMoveFromIdx = lastMoveCoords?.fromIdx ?? -1
  $: lastMoveToIdx = lastMoveCoords?.toIdx ?? -1

  // Heatmap overlay: only computed when the toggle is on AND the ply has data.
  // Promotion variants (e.g., "5g5f" and "5g5f+") share a destination — sum
  // their probabilities so the spectator sees one combined "considered going to 5f".
  $: heatmap = (() => {
    if (!$showcaseHeatmapEnabled || !move?.move_heatmap_json) return null
    const raw = safeParse(move.move_heatmap_json, null)
    if (!raw || typeof raw !== 'object') return null
    const out = {}
    for (const [usi, prob] of Object.entries(raw)) {
      const parsed = parseUsi(usi)
      if (!parsed) continue
      out[parsed.toIdx] = (out[parsed.toIdx] ?? 0) + prob
    }
    return out
  })()
</script>
```

Replace the `<Board ...>` line (line 44) with:

```svelte
          <Board
            board={board}
            inCheck={!!move?.in_check}
            currentPlayer={move?.current_player || 'black'}
            lastMoveFromIdx={lastMoveFromIdx}
            lastMoveToIdx={lastMoveToIdx}
            heatmap={heatmap}
          />
```

Replace the `<div class="game-header">` block (lines 32-40) with one that includes the heatmap toggle:

```svelte
      <div class="game-header">
        <span class="player black">{game.name_black} ({game.elo_black?.toFixed(0) ?? '?'})</span>
        <span class="vs">vs</span>
        <span class="player white">{game.name_white} ({game.elo_white?.toFixed(0) ?? '?'})</span>
        <span class="ply">Ply {game.total_ply}</span>
        <button
          class="heatmap-toggle"
          on:click={() => showcaseHeatmapEnabled.update(v => !v)}
          aria-pressed={$showcaseHeatmapEnabled}
          title="Toggle policy heatmap overlay"
        >
          Heatmap: {$showcaseHeatmapEnabled ? 'On' : 'Off'}
        </button>
        {#if game.status !== 'in_progress'}
          <span class="result">{game.status.replaceAll('_', ' ')}</span>
        {/if}
      </div>
```

Add a CSS rule for the toggle inside the `<style>` block (alongside the existing `.player`, `.vs`, etc. rules):

```css
  .heatmap-toggle { padding: 4px 10px; background: var(--bg-elevated, transparent); color: var(--text-primary); border: 1px solid var(--border); border-radius: 4px; font-size: 12px; cursor: pointer; }
  .heatmap-toggle[aria-pressed="true"] { background: var(--accent-gold); color: #000; border-color: var(--accent-gold); }
  .heatmap-toggle:focus-visible { outline: 2px solid var(--accent-teal); outline-offset: 2px; }
```

- [ ] **Step 2: Verify the build still compiles**

```bash
cd webui && npx vite build 2>&1 | tail -10
```

Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add webui/src/lib/ShowcaseView.svelte
git commit -m "feat(webui): showcase board renders last-move + heatmap toggle"
```

---

## Task 10: Frontend — drop dead `lastMoveIdx` in `App.svelte`

**Files:**
- Modify: `webui/src/App.svelte` (lines 53-61, line 204)

- [ ] **Step 1: Remove the dead IIFE**

Delete lines 53-61 of `webui/src/App.svelte`:

```svelte
  let thumbPanelHeight = 0

  $: lastMoveIdx = (() => {
    try {
      const history = safeParse(moveHistory, [])
      if (history.length === 0) return -1
      return -1
    } catch { return -1 }
  })()
```

Replace with just:

```svelte
  let thumbPanelHeight = 0
```

- [ ] **Step 2: Remove the prop from the training-tab Board call**

Delete the `lastMoveIdx={lastMoveIdx}` line (line 204) from the `<Board ...>` element. The new Board props (`lastMoveFromIdx`, `lastMoveToIdx`, `heatmap`) all default to nothing-rendered, so omitting them is correct for the training tab.

- [ ] **Step 3: Verify the build still compiles**

```bash
cd webui && npx vite build 2>&1 | tail -10
```

Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add webui/src/App.svelte
git commit -m "chore(webui): drop dead lastMoveIdx placeholder from training tab"
```

---

## Task 11: Manual verification

- [ ] **Step 1: Restart the showcase sidecar against a real model**

The sidecar is launched as a long-running process. From the project root, follow whichever process-management approach is current (typically a systemd unit or a tmux session — check the runbook). The key requirements:

```bash
# Confirm the maturin rebuild happened (Task 1):
cd shogi-engine/crates/shogi-gym && source .venv/bin/activate && python -c "from shogi_gym import SpectatorEnv; print(hasattr(SpectatorEnv, 'legal_moves_with_usi'))"
# Expected: True
```

Then restart the showcase runner so it picks up the new code.

- [ ] **Step 2: Open the webui showcase tab and confirm visuals**

Open the webui in a browser (port per project README/runbook). Pick two league entries and start a match.

Verify:
- The most recent move's destination square shows the existing teal-bordered yellow tint.
- The from-square shows a thin gold inset border (no fill).
- The "Heatmap: Off" toggle is visible in the game header.
- Click the toggle. It changes to "Heatmap: On" with gold background.
- Squares the moved piece could have legally gone to are shaded warm yellow with intensity proportional to the model's policy probability. A near-forced move shows one strong shade; a wide-open mid-game position shows multiple lighter shades.
- Toggle off; overlay disappears.
- Reload the page; toggle state persists.
- Drop moves (e.g., `P*5e` after captures accumulate) light up all legal drop destinations for that piece type.
- Pre-existing match plies (NULL `move_heatmap_json`) show no overlay even with the toggle on.

- [ ] **Step 3: If everything looks right, no further action**

If anything looks off, capture details and feed back to a triage iteration.

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Last-move from-square (subtle) + to-square (prominent) — Task 8 (CSS), Task 9 (wiring)
- ✅ Drops highlight only destination — `parseUsi` returns `fromIdx: null` for drops (Task 6); Board renders nothing for `lastMoveFromIdx === -1` (Task 8)
- ✅ Toggleable policy heatmap — Task 7 (store), Task 8 (rendering), Task 9 (toggle UI + parsing)
- ✅ Drops use the same heatmap mechanism — `build_heatmap` filters by `_move_prefix` which works for both `"7g"` and `"P*"` (Task 2)
- ✅ Promotion variants summed — explicit in Task 9's `heatmap` derivation (`out[parsed.toIdx] = (out[parsed.toIdx] ?? 0) + prob`)
- ✅ Pre-existing data renders gracefully — column nullable, frontend checks `move?.move_heatmap_json` (Tasks 3, 9)
- ✅ Schema migration — Task 3
- ✅ Maturin rebuild step — Task 1 step 4 + Task 11 step 1
- ✅ Removal of dead `lastMoveIdx` IIFE — Task 10
- ✅ Visual treatment matches spec table — Task 8 CSS

**Type/signature consistency:**
- `build_heatmap()` signature in Task 2 matches usage in Task 5 (`chosen_usi=`, `legal_with_usi=`, `probs=`).
- `write_showcase_move()` signature in Task 4 matches usage in Task 5 (`move_heatmap_json=` kwarg).
- `parseUsi` return shape `{fromIdx, toIdx, isDrop, dropPiece}` is consistent across Tasks 6, 9.
- `Board.svelte` props `lastMoveFromIdx`, `lastMoveToIdx`, `heatmap` consistent across Tasks 8, 9, 10.
- localStorage key `showcaseHeatmapEnabled` consistent in Task 7.

**Placeholder scan:** None.
