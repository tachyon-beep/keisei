# League Dashboard Overhaul — Design Spec

**Date:** 2026-04-06
**Status:** Draft
**Tracks:** filigree keisei-907b775ad0 (partial), new work for dropped data streams
**Svelte version:** 4 (Svelte 5 migration deferred)

## Problem

The league dashboard has three classes of issues:

1. **Dropped data streams**: Backend sends `historical_library`, `gauntlet_results`, and `transitions` in both `init` and `league_update` WebSocket messages (`app.py:215-217`, `305-307`). The frontend `ws.js:121-125` ignores all three — the handler only destructures `entries`, `results`, `elo_history`.

2. **Wrong headline metrics**: Stats banner shows generic structural metrics (pool size, match count, top rated, Elo range) from `leagueStats` derived store. The design spec (tiered-opponent-pool.md §9.2, §15.2) requires role-specific metrics: Frontier Benchmark Elo, Dynamic League Elo, Recent Challenge Score, Historical Gauntlet Score.

3. **Flat leaderboard without role structure**: `LeagueTable.svelte` sorts all active entries by composite `elo_rating`. Role badges exist (roleIcons.js) but serve as cosmetic labels on a flat list, not structural grouping. The spec (§15.1) requires grouping by role with slot fill visibility.

## Design Decisions

Informed by three expert reviews (UX, Architecture, Systems Thinking):

- **Bottom-left panel is tabbed, not replaced.** Recent Matches provides real-time liveness signal; Historical Library is slow-changing reference data. Both coexist as tabs within the same panel area.
- **Stats banner capped at 5 cards.** 6-7 cards compress values at typical viewport widths. Role-specific win rates (W10) move into EntryDetail instead.
- **Frontier Elo gets visual precedence.** Uses `stat-card.highlight` accent treatment. Without this, Dynamic League Elo (K=24) will dominate attention over Frontier Benchmark Elo (K=16) because it changes more dramatically.
- **EntryDetail renders below the table**, not as an inline row expansion. Two-section detail panel (Last Round + Overall Record) is too content-heavy for a `<tr>`.
- **`learnerEntry` derived store** eliminates prop drilling of `learnerName` through App → LeagueView → LeagueTable/MatchupMatrix.
- **Readable metric labels**, not abbreviations. "Frontier Elo" not "FBE".
- **Admission queue (W7) deferred** pending confirmation that queue depth data is available in the WebSocket payload. Not blocked on this.

## Architecture

### New Stores (`stores/league.js`)

```
historicalLibrary    — writable([])   — slot assignments from backend
gauntletResults      — writable([])   — gauntlet match records from backend
leagueTransitions    — writable([])   — role/status transitions from backend

learnerEntry         — derived        — entry matching trainingState.display_name (cross-store dep, see below)
leagueByRole         — derived        — groups leagueRanked entries by entry.role
transitionCounts     — derived        — {promotions, evictions, admissions} from leagueTransitions (see classification rules below)
```

**`learnerEntry` cross-store dependency:** This derived store requires importing `trainingState` from `stores/training.js` into `league.js` — a new coupling between two currently independent store modules. The `trainingState.display_name` is set via the `training_status` WebSocket message, which may arrive after the `init` message. The derived store must handle `null` gracefully:
```js
export const learnerEntry = derived(
  [leagueEntries, trainingState],
  ([$entries, $state]) => {
    const name = $state?.display_name
    if (!name) return null
    return $entries.find(e => e.display_name === name) || null
  }
)
```

**`leagueByRole` output contract:** Groups `leagueRanked` (active-only) entries by `entry.role`. Entries with unknown or null roles are placed in an `"other"` bucket. Retired entries are excluded (inherited from `leagueRanked`'s active filter). Output shape: `Map<string, Entry[]>` where keys are role strings.

**`transitionCounts` classification rules:**

| from_status | to_status | from_role | to_role | Category |
|-------------|-----------|-----------|---------|----------|
| `null`/missing | `active` | any | any | **admission** |
| `active` | `retired` | any | any | **eviction** |
| `active` | `active` | `recent_fixed` | `frontier_static` | **promotion** |
| `active` | `active` | `dynamic` | `frontier_static` | **promotion** |
| `active` | `active` | `dynamic` | `recent_fixed` | **promotion** |

All other transitions: ignored (not counted). Status transitions take precedence over role transitions — an entry that is both promoted and retired in the same record counts as eviction, not promotion.

### WebSocket Wiring (`lib/ws.js`)

Both `init` and `league_update` handlers updated to set:
- `historicalLibrary` from `msg.historical_library` (both message types use same key)
- `gauntletResults` from `msg.gauntlet_results` (both message types use same key)
- `leagueTransitions` from `msg.transitions` (both message types use same key)

Note: backend key is `transitions`, store name is `leagueTransitions` — mapping must be explicit in the handler.

All setters must use `|| []` fallback (matching existing pattern) to handle older backends or missing keys:
```js
historicalLibrary.set(msg.historical_library || [])
gauntletResults.set(msg.gauntlet_results || [])
leagueTransitions.set(msg.transitions || [])
```

Key name reference (init vs league_update differ for existing fields):
- init: `league_entries`, `league_results`, `elo_history`
- league_update: `entries`, `results`, `elo_history`
- Both: `historical_library`, `gauntlet_results`, `transitions` (same keys in both)

### Component Changes

**Modified:**
- `LeagueView.svelte` — new stats banner, tabbed bottom-left panel, below-table EntryDetail area
- `LeagueTable.svelte` — role-grouped toggle, protection indicator, expanded click targets EntryDetail below table
- `LeagueEventLog.svelte` — transition count summary header

**New:**
- `EntryDetail.svelte` — Last Round + Overall Record sections for selected entry
- `HistoricalLibrary.svelte` — slot table + gauntlet results display

**Unchanged:**
- `MatchupMatrix.svelte` — no changes
- `RecentMatches.svelte` — no changes (now lives in a tab)
- `MatchHistory.svelte` — deprecated but not deleted (replaced by EntryDetail)

### Data Flow

All data already exists in the WebSocket payload except `games_vs_*` counts.

**One backend change required:** `read_league_data()` in `db.py:398-403` must add `games_vs_frontier, games_vs_dynamic, games_vs_recent` to the SELECT query. These columns exist in the `league_entries` table (`db.py:108-110`) but are not included in the query output.

Entry fields already sent but unused by frontend:
- `elo_frontier`, `elo_dynamic`, `elo_recent`, `elo_historical` — role-specific Elo ratings
- `protection_remaining` — epochs of eviction protection left

Entry fields that exist in DB but are NOT yet sent (requires backend fix):
- `games_vs_frontier`, `games_vs_dynamic`, `games_vs_recent` — per-role game counts

Match result fields already sent:
- `role_a`, `role_b` — roles of participants (for role-specific win rate computation)
- `match_type` — type of match (frontier, dynamic, recent, gauntlet)

## Layout

### Target Layout

```
+------------------------------------------------------------+
| Frontier Elo* | Dynamic Elo | Challenge | Gauntlet | Pool   |
| (highlighted)  |             |  Score    |  Score   | 3/20   |
+---------------------------+--------------------------------+
| Elo Leaderboard           | Matchup Matrix (NxN)          |
|  [Flat | Grouped] toggle  |                               |
|                           |                               |
|  -- Frontier (3/5) ----  |                               |
|  #1  entry  1200  ...    |                               |
|  #2  entry  1150  ...    |                               |
|  -- Recent (5/5) ------  |                               |
|  #3  entry  1100  ...    | Elo Over Time Chart           |
|  ...                     |                               |
|  -- Dynamic (7/10) ----  |                               |
|  ...                     |                               |
+---------------------------+                               |
| EntryDetail (when entry   |                               |
|  is selected):            |                               |
| Last Round | Overall      |                               |
+-------------+-------------+                               |
| Event Log   | [Recent Matches | Hist. Library]            |
| (+ counts)  |  (tabbed)                                   |
+-------------+-------------+-------------------------------+
```

### Stats Banner (5 cards)

| Card | Label | Source | Visual |
|------|-------|--------|--------|
| 1 | Frontier Elo | `learnerEntry.elo_frontier` | `highlight` class (accent border + bg) |
| 2 | League Elo | `learnerEntry.elo_dynamic` | default |
| 3 | Challenge | `learnerEntry.elo_recent` | default |
| 4 | Gauntlet | `learnerEntry.elo_historical` | default |
| 5 | Pool | `leagueStats.poolSize` / totalSlots | default |

All cards show the value prominently with a readable label below. No abbreviations.

### Role-Grouped Leaderboard

Toggle: `[Flat | Grouped]` using `role="radiogroup" aria-label="Leaderboard view"` with each option as `role="radio" aria-checked="true|false"`. This correctly conveys mutual exclusion to screen readers (unlike `aria-pressed` which implies independent toggles). Toggle state is local component state — does not persist across page loads or reconnects.

When grouped:
- Section headers rendered as `<tr class="group-header"><th colspan="N" scope="colgroup">` to preserve table ARIA grid structure. Icon in `<span aria-hidden="true">`, label as plain text. Example: `"🛡 Frontier · 3/5"`
- Per-row role icons suppressed within their own group section (reduces redundancy)
- Sorting works within each group
- Section capacity denominators: hardcoded from pool config constants (5/5/10) — these are stable architectural limits, not dynamic values

When flat: current behavior preserved (sorted by selected column, role badges on every row).

Protection indicator: entries with `protection_remaining > 0` show a small `🛡 N` badge after name, where N is epochs remaining. Badge uses `aria-label="Protected for N epochs"` with emoji in `aria-hidden="true"`.

Switching between Flat and Grouped views preserves the current `focusedEntryId`; EntryDetail remains visible if the selected entry is present in the new view.

### EntryDetail (below-table panel)

Appears when a leaderboard row is clicked. Replaces the inline `MatchHistory` row expansion. Phase 4 removes the `MatchHistory` import from `LeagueTable.svelte` and the inline `expandedId` row expansion logic.

**Layout constraints:** `max-height: 200px` with `overflow-y: auto`. This prevents the panel from pushing the bottom row (Event Log + tabbed panel) off-screen on monitors under 900px.

**Accessibility:** Panel has a visually-hidden `<h3>` heading with the entry name for screen reader navigation. On row activation (click or Enter), focus moves to the EntryDetail heading. Panel uses `aria-live="polite"` so content changes are announced.

**Section A — Last Round:**
- Filters `leagueResults` where epoch equals the entry's most recent match epoch
- Shows each opponent: name, role badge, W/L/D for that round, Elo delta
- Empty state: "No matches in the current round"

**Section B — Overall Record:**
- Uses `headToHead` store filtered to selected entry's pairings
- Shows each opponent: total W/L/D, win rate %, total games
- Sorted by total games descending (most-played first)
- Also shows role-specific Elo breakdown and `games_vs_*` counts from the entry data (addresses W10)

**Section C — Role-Specific Stats:**
- Mini stat row: Frontier Elo / Dynamic Elo / Recent Elo / Historical Elo
- Games by tier: vs Frontier / vs Dynamic / vs Recent

Click on leaderboard row toggles the detail panel. `focusedEntryId` store continues to drive matrix cross-highlighting.

### Tabbed Bottom-Left Panel

Implements the full ARIA tab pattern: `role="tablist"` container, `role="tab"` on each tab button, `role="tabpanel"` on the content area. Arrow keys move between tabs, `Tab` exits the tab list into the panel content. **Default tab: Recent Matches** (preserves the liveness monitoring signal on initial load and reconnect).

### Historical Library Panel (tab)

Lives in tabbed panel alongside Recent Matches in the bottom-left area.

**Slot Table:**
- Columns: Slot #, Entry Name, Target Epoch, Actual Epoch, Selection Mode
- All columns use `<th scope="col">`; table has `<caption class="sr-only">Historical library slot assignments</caption>`
- Source: `historicalLibrary` store
- Empty state: "No historical slots configured"

**Latest Gauntlet Results:**
- Grouped by epoch (most recent first)
- Per row: entry vs historical slot entry, W/L/D, Elo delta
- Source: `gauntletResults` store
- Staleness indicator: "Last gauntlet: N epochs ago" computed from max gauntlet epoch vs current training epoch

### Event Log Enrichment

Summary line at top of event log: `"↑ 3 promoted · ↓ 2 evicted · → 5 admitted"`. Uses `aria-live="polite"` so screen readers announce count changes (these are ambient status updates, not errors).

Counts derived from `transitionCounts` store (real DB transitions from `leagueTransitions`, not client-side diff).

Batch-collapse during bursts: if multiple transitions of the same type occur in the same epoch, show as single line (e.g., "↓ 3 evictions in epoch 847") rather than 3 separate events.

## Requirements Coverage

| Req | Description | Resolution | Phase |
|-----|-------------|------------|-------|
| W1 | Role-grouped leaderboard | Grouped view with section headers | 3 |
| W2 | Historical Library panel | Tab in bottom-left panel | 5 |
| W3 | Frontier Benchmark Elo | Banner card 1 (highlighted) | 2 |
| W4 | Dynamic League Elo | Banner card 2 | 2 |
| W5 | Recent Challenge Score | Banner card 3 | 2 |
| W6 | Historical Gauntlet Score | Banner card 4 | 2 |
| W7 | Dynamic admission queue | Deferred (data availability unconfirmed) | — |
| W8 | Eviction/promotion counts | Event log summary line | 6 |
| W9 | Historical milestone epochs | Historical Library slot table | 5 |
| W10 | Role-specific win rates | EntryDetail Section C | 4 |
| W11 | Role badges | Already DONE | — |
| W12 | No flat without markers | Already DONE | — |
| W13 | Show retired entries | Already DONE | — |
| W14 | Protection window indicators | Shield badge on protected entries | 3 |
| W15 | Svelte 5 migration | SKIPPED (deferred) | — |

## Phases

```
Phase 1 (stores + ws wiring) ──→ Phase 2 (stats banner)
                              ──→ Phase 3 (role grouping)
                              ──→ Phase 4 (EntryDetail)
                              ──→ Phase 5 (Historical Library)
                              ──→ Phase 6 (Event Log)
```

Phase 1 is the only prerequisite. Phases 2-6 are independent of each other.

## Testing

### Store tests (`league.test.js`)

New derived store tests (3 new stores must be added to `beforeEach` reset block):

**`learnerEntry`:**
- Returns `null` when `trainingState` is null
- Returns `null` when `display_name` is set but no entry matches
- Returns matching entry when `display_name` matches

**`leagueByRole`:**
- Groups active entries by role correctly
- Places entries with unknown/null roles in `"other"` bucket
- Excludes retired entries (inherited from `leagueRanked` active filter)

**`transitionCounts`:**
- Counts admissions (`to_status: active` from null/missing)
- Counts evictions (`from_status: active, to_status: retired`)
- Counts promotions (role upgrades while remaining active)
- Ignores unrecognised transition types (count stays 0)

### WebSocket tests (`ws.test.js`)

- Verify 3 new stores set on both `init` and `league_update` message types
- **Explicitly test key mapping**: send `{ transitions: [...] }` and assert `leagueTransitions` store is non-empty (this is the exact bug class being fixed — silent key mismatch)
- Test `|| []` fallback: send messages with each new key absent, verify stores default to `[]` not `undefined`
- Apply `vi.resetModules()` pattern if tests assert on `leagueEvents` side effects

### Component tests (new — `@testing-library/svelte`)

Minimum component test coverage for new/modified components:

**`EntryDetail`:**
- Renders empty state when no entry selected
- Renders "No matches in the current round" when entry has no results for latest epoch
- Renders Last Round and Overall Record sections with mock data

**`LeagueTable` (grouped toggle):**
- Flat view: entries sorted by Elo (existing behavior preserved)
- Grouped view: entries grouped by role with section headers
- Toggle preserves `focusedEntryId`

### Phase 6 specific

- Batch-collapse logic: unit test for epoch grouping (multiple same-type transitions in one epoch → single collapsed line)

## Known Limitations / Future Work

- **Dynamic pool quality**: Slot fill ratio doesn't show lineage diversity or competitive strength within tier (systems thinking finding)
- **Recent Fixed calibration debt**: No visibility into whether Recent entries have enough games for promotion review
- **Gauntlet timing**: Staleness indicator added, but no "next gauntlet scheduled" display
- **`diffLeagueEntries` module-level state**: Architecture reviewer flagged this as fragile; Phase 6 should avoid adding cross-references to it. If scope allows, refactor to pass previous state explicitly. Note: on WebSocket reconnect, `init` calls `diffLeagueEntries` with the full entry set, generating spurious "arrival" events — consider clearing `_prevEntryMap` in the `init` handler path.
- **Dual provenance in Event Log**: Summary counts come from backend `transitionCounts`; individual event items come from client-side `diffLeagueEntries`. These may temporarily disagree during high-churn epochs. Accepted limitation — Phase 6 must not make this worse.
- **Attention loop residual**: Frontier Elo highlight slows but does not fully break the R1 attention loop. Dynamic Elo's higher K-factor (24 vs 16) produces more dramatic per-epoch changes, which may still dominate attention over time. Consider adding metric tooltips explaining what each Elo measures and what direction is healthy.
- **Slot capacity denominators**: Hardcoded 5/5/10 from pool config. If pool tier sizes change in config, frontend must be updated manually. Future enhancement: send tier capacity limits in `init` payload.
