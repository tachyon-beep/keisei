# Tournament/League UI for Keisei Training Dashboard

**Date:** 2026-04-02
**Status:** Draft
**Scope:** WebUI frontend + backend WebSocket pipeline for league/tournament visibility

## Problem

The Keisei training dashboard monitors self-play training in real time (game viewer, metrics charts, status bar). The backend already supports opponent-pool league training with Elo tracking (`league_entries`, `league_results`, `OpponentPool`, `OpponentSampler`), but none of this data is visible in the UI. Users cannot see who the learner is playing against, how Elo ratings evolve, or review match history.

## Goals

1. Surface league standings, match history, and Elo trends in the dashboard
2. Show opponent identity during league matches in the training view
3. Add navigation to switch between training monitoring and league analysis
4. Improve training view layout density (compact thumbnails, player cards, expandable charts)

## Non-Goals

- Tournament bracket visualization (round-robin/elimination formats)
- Interactive opponent pool management (pin/unpin/evict from UI)
- Mobile-first responsive redesign (maintain existing mobile breakpoints)

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Navigation model | Tab bar (conditional rendering) | 2 views, single-user dashboard — no router dependency needed |
| Elo history storage | New `elo_history` table | Clean slate (no migration), simpler than replaying from results |
| League data poll cadence | 5s (vs 0.2s for games/metrics) | League data changes per-epoch, not per-step |
| League data delivery | Full replacement per message | Pool is small (<20 entries) — diffing not worth the complexity |
| Player identity display | Dedicated player cards between thumbnails and board | Prominent, always visible, richer than inline text badge |
| Chart interaction | Mini sparklines with click-to-expand | Compact default, detail on demand — mirrors thumbnail pattern |

---

## 1. Navigation

### Tab Bar

A horizontal tab toggle integrated into the right side of the status header bar. Two tabs: **Training** (default) | **League**.

**State management:** New store `stores/navigation.js` with a single `writable('training')`. `App.svelte` conditionally renders the training view or league view based on this store.

**Behavior:**
- Active tab has accent border and background; inactive tab has subtle border
- Tab state resets to "training" on page load
- Both views receive WebSocket data regardless of which tab is active (stores update independently of visibility)
- Keyboard accessible: tabs are `<button>` elements with `aria-selected`

### Component: `TabBar.svelte`

Renders inside `StatusIndicator.svelte` on the right side (replacing the current player-name-only right section). Player name moves into the learner card.

Props: none (reads from `activeTab` store directly).

---

## 2. Backend Data Pipeline

### New Table: `elo_history`

Added to `init_db()` in `keisei/db.py`:

```sql
CREATE TABLE IF NOT EXISTS elo_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id    INTEGER NOT NULL REFERENCES league_entries(id),
    epoch       INTEGER NOT NULL,
    elo_rating  REAL NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_elo_history_entry ON elo_history(entry_id);
```

### OpponentPool Change

`OpponentPool.update_elo()` in `keisei/training/league.py` writes a row to `elo_history` after updating `league_entries.elo_rating`. Single additional INSERT in the same connection.

When `_delete_entry()` evicts a pool entry, it also deletes the corresponding `elo_history` rows (CASCADE-like cleanup in application code, consistent with existing `league_results` cleanup).

### WebSocket: `league_update` Message

New message type sent by `_poll_and_push()` in `keisei/server/app.py`.

**Poll cadence:** Every 5 seconds (separate counter from the 0.2s game/metrics tick).

**Change detection:** Track `last_league_entry_count` and `last_league_result_id`. Only send `league_update` when either changes. On first connection, always send (as part of `init`).

**Message shape:**

```json
{
  "type": "league_update",
  "entries": [
    {
      "id": 1,
      "architecture": "transformer",
      "elo_rating": 1247.0,
      "games_played": 86,
      "created_epoch": 12,
      "created_at": "2026-04-02T10:00:00Z"
    }
  ],
  "results": [
    {
      "id": 1,
      "epoch": 12,
      "learner_id": 1,
      "opponent_id": 2,
      "wins": 6,
      "losses": 2,
      "draws": 2,
      "recorded_at": "2026-04-02T10:05:00Z"
    }
  ],
  "elo_history": [
    { "entry_id": 1, "epoch": 10, "elo_rating": 1180.0 },
    { "entry_id": 1, "epoch": 12, "elo_rating": 1247.0 }
  ]
}
```

### Init Message Extension

The existing `init` message gains three new fields: `league_entries`, `league_results`, `elo_history` — same shapes as above.

### Game Snapshot Extension

The `game_update` message already sends `game_snapshots` rows. Two columns exist in the schema but are not currently included in the WebSocket payload: `game_type` and `demo_slot`. The server query in `_poll_and_push()` will include these, plus a new `opponent_id` field.

`opponent_id` is nullable — it is only set when `game_type` is a league match. The training loop is responsible for writing this value to the `game_snapshots` row when starting a league game.

**Schema change to `game_snapshots`:** Add `opponent_id INTEGER REFERENCES league_entries(id)` column. Added in `init_db()` (no migration).

---

## 3. Frontend Stores

### `stores/league.js`

```javascript
// Writable stores
leagueEntries   // Array of entry objects
leagueResults   // Array of result objects  
eloHistory      // Array of { entry_id, epoch, elo_rating }

// Derived stores
leagueRanked    // leagueEntries sorted by elo_rating DESC, with .rank injected
```

### `stores/navigation.js`

```javascript
activeTab       // writable('training') — values: 'training' | 'league'
```

### Derived: `selectedOpponent`

In `stores/games.js`, a new derived store that cross-references `selectedGame.opponent_id` against `leagueEntries` to produce `{ architecture, elo_rating, games_played }` or `null`.

### WebSocket Handler

`ws.js` `handleMessage()` gains:
- `init` case: set `leagueEntries`, `leagueResults`, `eloHistory` from payload
- New `league_update` case: full replacement of all three league stores

---

## 4. Training Tab — Revised Layout

### Structure

```
┌──────────── Status Bar ──── [Training | League] ──┐
├────────┬────────────┬─────────────────────────────┤
│ 4×4    │ ☗ Learner  │ Board + Eval + Info + Moves  │
│ thumb  │    VS      │                              │
│ grid   │ ☖ Opponent │                              │
│        │            │                              │
│  (all three columns equal height)                  │
├───────────────────────────────────────────────────┤
│ Mini₁  │  Mini₂  │  Mini₃  │  Mini₄              │
│ (click any mini chart to expand it to full detail) │
└───────────────────────────────────────────────────┘
```

### Thumbnail Grid

- Grid changes from `repeat(8, 1fr)` to `repeat(4, 1fr)`
- `$games.slice(0, 16)` instead of `slice(0, 32)`
- Panel width shrinks proportionally
- Click behavior unchanged: promotes game to main board view

### Player Cards — `PlayerCard.svelte`

A single component rendered twice (once for learner, once for opponent).

**Props:**
- `role`: `'learner'` | `'opponent'`
- `name`: display name / architecture string
- `elo`: number or null
- `detail`: secondary text (e.g., "Epoch 12 · 48.2k steps" or "124 games played")

**Layout:** Compact card with role icon (☗/☖), name, Elo badge, detail line. Learner uses `--accent-green`, opponent uses `--accent-blue`.

**Data sources:**
- Learner card: `trainingState` store (display_name, model_arch, epoch, step) + learner Elo from the most recent league entry matching the current architecture and epoch (highest `created_epoch` in `leagueEntries`), or no Elo shown if pool is empty
- Opponent card: `selectedOpponent` derived store, or fallback text "Self-play" when null

**VS separator:** A small centered "VS" text between the two cards (not a separate component — just markup in `App.svelte`).

### Metrics — Mini Sparklines with Click-to-Expand

`MetricsGrid.svelte` revised:

**Default state:** 4 charts in a single `1fr 1fr 1fr 1fr` grid row, each ~150px tall. Minimal: title, sparkline, no axes/legend. Shows trend shape at a glance.

**Expanded state:** When a mini chart is clicked, it expands to a full-width detail view (~300px tall) with axes, legend, hover tooltip. The other three remain as minis. Clicking the expanded chart collapses it. Clicking a different mini swaps which one is expanded.

**State:** Local component state `expandedChart` (index or null). No store needed — this is ephemeral UI state.

**Chart rendering:** Mini charts use `MetricsChart.svelte` with reduced `height` prop and a `compact` boolean prop that suppresses legend/annotation. Expanded chart uses full height with legend/annotation.

---

## 5. League Tab — `LeagueView.svelte`

### Layout

Top half: `LeagueTable.svelte` (leaderboard)
Bottom half: Elo trend chart (reuses `MetricsChart.svelte`)

### `LeagueTable.svelte`

Sortable table of pool entries.

**Columns:** Rank | Model | Elo | Games | Epoch
- Default sort: Elo descending
- Click column header to change sort column and toggle asc/desc
- Current best (rank 1) highlighted with `--accent-green`

**Row click:** Toggles inline `MatchHistory.svelte` expansion below that row, showing all matches involving the selected entry.

**Empty state:** "No league entries yet. League data appears once opponent pool training begins."

**Styling:** Follows existing dashboard conventions — dark theme, 12px uppercase section headers, `--border` separators, monospace for numbers.

### `MatchHistory.svelte`

Rendered inline below a selected leaderboard row.

**Columns:** Epoch | Opponent | W | L | D
- Opponent column shows architecture name (looked up from entries by opponent_id)
- W/L/D cells color-coded: green for wins, red for losses, amber for draws
- Sorted by epoch descending (most recent first)
- Scrollable if many results (max-height with overflow-y)

### Elo Trend Chart

Reuses `MetricsChart.svelte` with:
- `xData`: epoch values from `eloHistory`
- `series`: one line per `entry_id`, grouped from `eloHistory`
- Colors cycle through `--accent-green`, `--accent-blue`, `--accent-amber`, `--accent-purple`, `--accent-pink`
- Legend shows architecture name + current Elo

New helper `lib/eloChartData.js` to transform the flat `eloHistory` array into the grouped `{ xData, series }` shape that `MetricsChart` expects.

---

## 6. Accessibility

### New ARIA Patterns

- Tab bar: `role="tablist"` with `role="tab"` buttons and `aria-selected`
- League table: standard `<table>` with `<thead>`/`<tbody>`, sortable columns indicated with `aria-sort`
- Match history expansion: `aria-expanded` on the triggering row
- Mini chart expand/collapse: `aria-expanded` on the chart wrapper
- Player cards: `aria-label` describing the full card content

### Live Regions

- Status bar stats div gets `aria-live="polite"` (training status changes)
- League table update does not need live region (user-initiated tab switch)

---

## 7. File Inventory

### New Files

| File | Purpose |
|------|---------|
| `webui/src/stores/navigation.js` | `activeTab` store |
| `webui/src/stores/league.js` | League entries, results, elo history stores |
| `webui/src/lib/TabBar.svelte` | Tab toggle component |
| `webui/src/lib/PlayerCard.svelte` | Learner/opponent identity card |
| `webui/src/lib/LeagueView.svelte` | League tab container |
| `webui/src/lib/LeagueTable.svelte` | Sortable Elo leaderboard |
| `webui/src/lib/MatchHistory.svelte` | Inline match results |
| `webui/src/lib/eloChartData.js` | Transform elo_history into chart series |
| `webui/src/lib/eloChartData.test.js` | Tests for chart data helper |

### Modified Files

| File | Changes |
|------|---------|
| `keisei/db.py` | Add `elo_history` table, add `opponent_id` to `game_snapshots` |
| `keisei/training/league.py` | `update_elo()` writes to `elo_history`, `_delete_entry()` cleans up history |
| `keisei/server/app.py` | League data polling (5s cadence), `league_update` message, extended `init` |
| `webui/src/lib/ws.js` | Handle `league_update`, extend `init` handler |
| `webui/src/stores/games.js` | Add `selectedOpponent` derived store |
| `webui/src/App.svelte` | Tab routing, 3-column layout, player cards, revised metrics |
| `webui/src/lib/StatusIndicator.svelte` | Integrate `TabBar`, move player name to card |
| `webui/src/lib/MetricsGrid.svelte` | 4x1 mini sparklines with click-to-expand |
| `webui/src/lib/MetricsChart.svelte` | Add `compact` prop for mini mode |
| `webui/src/lib/GameThumbnail.svelte` | No code change, just grid CSS in parent |
| `webui/src/app.css` | Tab bar styles, player card variables |

---

## 8. Testing Strategy

### Unit Tests (Vitest)

- `eloChartData.test.js`: grouping, empty data, single entry, multiple entries with gaps
- `stores/league.js`: `leagueRanked` derived sort correctness, rank injection
- `stores/games.js`: `selectedOpponent` lookup with match, without match, null opponent_id
- `ws.js`: `handleMessage` for `league_update` and extended `init`

### Backend Tests (pytest)

- `elo_history` table creation and writes via `update_elo()`
- `_delete_entry()` cascades to `elo_history`
- `_poll_and_push()` league query at 5s cadence (mock timing)
- `league_update` message shape validation
- `game_snapshots` includes `opponent_id` when set

### Manual Verification

- Tab switching preserves WebSocket connection and store state
- Mini chart expand/collapse animation
- League table sort by each column
- Match history inline expansion
- Player cards update when switching games (self-play vs league)
- Empty states: no league data, no games, no metrics
