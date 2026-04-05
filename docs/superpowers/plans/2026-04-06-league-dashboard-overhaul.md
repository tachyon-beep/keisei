# League Dashboard Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire 3 dropped WebSocket data streams, replace generic stats with role-specific metrics, add role-grouped leaderboard, rich entry detail panel, historical library tab, and event log enrichment.

**Architecture:** 6 phases — Phase 1 (stores + WS wiring) is prerequisite; Phases 2-6 are independent. All frontend changes are Svelte 4. One backend change (add `games_vs_*` to SELECT query). No new npm dependencies.

**Tech Stack:** Svelte 4, Vitest, Python/SQLite (backend), WebSocket

**Spec:** `docs/superpowers/specs/2026-04-06-league-dashboard-overhaul-design.md`

---

## File Map

**Backend (Python):**
- Modify: `keisei/db.py:397-403` — add `games_vs_*` columns to SELECT
- Test: `tests/test_db.py` — verify new columns in output

**Frontend stores:**
- Modify: `webui/src/stores/league.js` — add 3 writable + 3 derived stores
- Test: `webui/src/stores/league.test.js` — tests for new derived stores

**Frontend WebSocket:**
- Modify: `webui/src/lib/ws.js` — wire 3 new stores in init + league_update
- Test: `webui/src/lib/ws.test.js` — verify key mapping and fallbacks

**Frontend components:**
- Modify: `webui/src/lib/LeagueView.svelte` — new banner, tabbed bottom panel, EntryDetail area
- Modify: `webui/src/lib/LeagueTable.svelte` — role-grouped toggle, protection badge, wire EntryDetail
- Modify: `webui/src/lib/LeagueEventLog.svelte` — transition count summary
- Create: `webui/src/lib/EntryDetail.svelte` — last round + overall record panel
- Create: `webui/src/lib/HistoricalLibrary.svelte` — slot table + gauntlet results

---

## Task 1: Backend — Add `games_vs_*` to SELECT query

**Files:**
- Modify: `keisei/db.py:397-403`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_db.py` in class `TestLeagueDataReaders`:

```python
def test_read_league_data_includes_games_vs_counts(self, tmp_path):
    import sqlite3
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO league_entries (architecture, model_params, checkpoint_path, created_epoch, "
        "games_vs_frontier, games_vs_dynamic, games_vs_recent) "
        "VALUES ('transformer', '{}', '/tmp/ckpt.pt', 5, 10, 20, 30)"
    )
    conn.commit()
    conn.close()
    data = read_league_data(db_path)
    entry = data["entries"][0]
    assert entry["games_vs_frontier"] == 10
    assert entry["games_vs_dynamic"] == 20
    assert entry["games_vs_recent"] == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py::TestLeagueDataReaders::test_read_league_data_includes_games_vs_counts -v`
Expected: FAIL with `KeyError: 'games_vs_frontier'`

- [ ] **Step 3: Add columns to SELECT**

In `keisei/db.py`, change the SELECT in `read_league_data` (line 397-403):

```python
        entries = conn.execute(
            "SELECT id, display_name, flavour_facts, model_params, architecture, "
            "elo_rating, games_played, created_epoch, created_at, "
            "role, status, parent_entry_id, lineage_group, protection_remaining, last_match_at, "
            "elo_frontier, elo_dynamic, elo_recent, elo_historical, "
            "optimizer_path, update_count, last_train_at, "
            "games_vs_frontier, games_vs_dynamic, games_vs_recent "
            "FROM league_entries ORDER BY elo_rating DESC"
        ).fetchall()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db.py::TestLeagueDataReaders -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/db.py tests/test_db.py
git commit -m "feat: include games_vs_* counts in read_league_data SELECT"
```

---

## Task 2: Stores — Add 3 writable stores and `learnerEntry` derived store

**Files:**
- Modify: `webui/src/stores/league.js`
- Test: `webui/src/stores/league.test.js`

- [ ] **Step 1: Write failing tests for new writable stores and `learnerEntry`**

Add imports and reset to `webui/src/stores/league.test.js`. Update the top import:

```js
import {
  leagueEntries, leagueResults, eloHistory,
  leagueRanked, entryWLD, headToHead, eloDelta, leagueStats,
  historicalLibrary, gauntletResults, leagueTransitions,
  learnerEntry,
} from './league.js'
import { trainingState } from './training.js'
```

Update `beforeEach`:

```js
beforeEach(() => {
  leagueEntries.set([])
  leagueResults.set([])
  eloHistory.set([])
  historicalLibrary.set([])
  gauntletResults.set([])
  leagueTransitions.set([])
  trainingState.set(null)
})
```

Add test block:

```js
describe('learnerEntry', () => {
  it('returns null when trainingState is null', () => {
    // trainingState defaults to null — no entries match
    expect(get(learnerEntry)).toBeNull()
  })

  it('returns null when display_name is set but no entry matches', () => {
    trainingState.set({ display_name: 'NonExistent' })
    leagueEntries.set([
      { id: 1, display_name: 'Bot-A', elo_rating: 1000, status: 'active' },
    ])
    expect(get(learnerEntry)).toBeNull()
  })

  it('returns matching entry when display_name matches', () => {
    trainingState.set({ display_name: 'Bot-A' })
    leagueEntries.set([
      { id: 1, display_name: 'Bot-A', elo_rating: 1000, status: 'active' },
      { id: 2, display_name: 'Bot-B', elo_rating: 1100, status: 'active' },
    ])
    const entry = get(learnerEntry)
    expect(entry).not.toBeNull()
    expect(entry.id).toBe(1)
    expect(entry.display_name).toBe('Bot-A')
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd webui && npx vitest run src/stores/league.test.js`
Expected: FAIL — `historicalLibrary` is not exported

- [ ] **Step 3: Add writable stores and `learnerEntry` to `league.js`**

Add at the top of `webui/src/stores/league.js`, after the existing writable declarations (after line 8):

```js
import { trainingState } from './training.js'

export const historicalLibrary = writable([])
export const gauntletResults = writable([])
export const leagueTransitions = writable([])

/** The league entry matching the current learner (by display_name from trainingState) */
export const learnerEntry = derived(
  [leagueEntries, trainingState],
  ([$entries, $state]) => {
    const name = $state?.display_name
    if (!name) return null
    return $entries.find(e => e.display_name === name) || null
  }
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd webui && npx vitest run src/stores/league.test.js`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add webui/src/stores/league.js webui/src/stores/league.test.js
git commit -m "feat: add historicalLibrary, gauntletResults, leagueTransitions stores + learnerEntry derived"
```

---

## Task 3: Stores — Add `leagueByRole` and `transitionCounts` derived stores

**Files:**
- Modify: `webui/src/stores/league.js`
- Test: `webui/src/stores/league.test.js`

- [ ] **Step 1: Write failing tests for `leagueByRole`**

Add to `webui/src/stores/league.test.js`:

```js
import {
  // ... existing imports ...,
  leagueByRole, transitionCounts,
} from './league.js'

describe('leagueByRole', () => {
  it('returns empty map when no entries', () => {
    expect(get(leagueByRole).size).toBe(0)
  })

  it('groups active entries by role', () => {
    leagueEntries.set([
      { id: 1, role: 'frontier_static', elo_rating: 1200, status: 'active' },
      { id: 2, role: 'frontier_static', elo_rating: 1100, status: 'active' },
      { id: 3, role: 'dynamic', elo_rating: 1000, status: 'active' },
    ])
    const byRole = get(leagueByRole)
    expect(byRole.get('frontier_static')).toHaveLength(2)
    expect(byRole.get('dynamic')).toHaveLength(1)
  })

  it('places entries with null/unknown role in "other" bucket', () => {
    leagueEntries.set([
      { id: 1, role: null, elo_rating: 1000, status: 'active' },
      { id: 2, role: 'some_future_role', elo_rating: 900, status: 'active' },
    ])
    const byRole = get(leagueByRole)
    expect(byRole.get('other')).toHaveLength(2)
  })

  it('excludes retired entries (inherited from leagueRanked)', () => {
    leagueEntries.set([
      { id: 1, role: 'dynamic', elo_rating: 1000, status: 'active' },
      { id: 2, role: 'dynamic', elo_rating: 900, status: 'retired' },
    ])
    const byRole = get(leagueByRole)
    expect(byRole.get('dynamic')).toHaveLength(1)
  })
})
```

- [ ] **Step 2: Write failing tests for `transitionCounts`**

```js
describe('transitionCounts', () => {
  it('returns zeros when no transitions', () => {
    const counts = get(transitionCounts)
    expect(counts).toEqual({ promotions: 0, evictions: 0, admissions: 0 })
  })

  it('counts admissions (null from_status to active)', () => {
    leagueTransitions.set([
      { id: 1, from_status: null, to_status: 'active', from_role: null, to_role: 'dynamic' },
    ])
    expect(get(transitionCounts).admissions).toBe(1)
  })

  it('counts admissions (from_status key absent)', () => {
    leagueTransitions.set([
      { id: 1, to_status: 'active', from_role: null, to_role: 'dynamic' },
    ])
    expect(get(transitionCounts).admissions).toBe(1)
  })

  it('counts evictions (active to retired)', () => {
    leagueTransitions.set([
      { id: 1, from_status: 'active', to_status: 'retired', from_role: 'dynamic', to_role: 'dynamic' },
    ])
    expect(get(transitionCounts).evictions).toBe(1)
  })

  it('counts promotions (role upgrade while active)', () => {
    leagueTransitions.set([
      { id: 1, from_status: 'active', to_status: 'active', from_role: 'dynamic', to_role: 'frontier_static' },
      { id: 2, from_status: 'active', to_status: 'active', from_role: 'recent_fixed', to_role: 'frontier_static' },
      { id: 3, from_status: 'active', to_status: 'active', from_role: 'dynamic', to_role: 'recent_fixed' },
    ])
    expect(get(transitionCounts).promotions).toBe(3)
  })

  it('ignores unrecognised transition types', () => {
    leagueTransitions.set([
      { id: 1, from_status: 'active', to_status: 'active', from_role: 'dynamic', to_role: 'dynamic' },
      { id: 2, from_status: 'retired', to_status: 'active', from_role: 'dynamic', to_role: 'dynamic' },
    ])
    const counts = get(transitionCounts)
    expect(counts).toEqual({ promotions: 0, evictions: 0, admissions: 0 })
  })

  it('status transitions take precedence over role transitions', () => {
    // Entry promoted AND retired in same record — counts as eviction
    leagueTransitions.set([
      { id: 1, from_status: 'active', to_status: 'retired', from_role: 'dynamic', to_role: 'frontier_static' },
    ])
    const counts = get(transitionCounts)
    expect(counts.evictions).toBe(1)
    expect(counts.promotions).toBe(0)
  })
})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd webui && npx vitest run src/stores/league.test.js`
Expected: FAIL — `leagueByRole` and `transitionCounts` not exported

- [ ] **Step 4: Implement `leagueByRole` and `transitionCounts`**

Add to `webui/src/stores/league.js`, after the `learnerEntry` definition:

```js
const KNOWN_ROLES = new Set(['frontier_static', 'recent_fixed', 'dynamic', 'historical'])

/** Groups active entries by role. Unknown/null roles go to 'other'. */
export const leagueByRole = derived(leagueRanked, ($ranked) => {
  const map = new Map()
  for (const entry of $ranked) {
    const key = KNOWN_ROLES.has(entry.role) ? entry.role : 'other'
    if (!map.has(key)) map.set(key, [])
    map.get(key).push(entry)
  }
  return map
})

const PROMOTION_PAIRS = new Set([
  'recent_fixed->frontier_static',
  'dynamic->frontier_static',
  'dynamic->recent_fixed',
])

/** Counts promotions, evictions, and admissions from backend transition records. */
export const transitionCounts = derived(leagueTransitions, ($transitions) => {
  let promotions = 0, evictions = 0, admissions = 0
  for (const t of $transitions) {
    // Status transitions take precedence
    if (!t.from_status && t.to_status === 'active') {
      admissions++
    } else if (t.from_status === 'active' && t.to_status === 'retired') {
      evictions++
    } else if (
      t.from_status === 'active' && t.to_status === 'active' &&
      PROMOTION_PAIRS.has(`${t.from_role}->${t.to_role}`)
    ) {
      promotions++
    }
  }
  return { promotions, evictions, admissions }
})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd webui && npx vitest run src/stores/league.test.js`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add webui/src/stores/league.js webui/src/stores/league.test.js
git commit -m "feat: add leagueByRole and transitionCounts derived stores"
```

---

## Task 4: WebSocket — Wire 3 new stores in `init` and `league_update`

**Files:**
- Modify: `webui/src/lib/ws.js`
- Test: `webui/src/lib/ws.test.js`

- [ ] **Step 1: Write failing tests for new store wiring**

Add imports to `webui/src/lib/ws.test.js`:

```js
import {
  leagueEntries, leagueResults, eloHistory,
  historicalLibrary, gauntletResults, leagueTransitions,
} from '../stores/league.js'
```

Update `beforeEach` to reset the 3 new stores:

```js
beforeEach(() => {
  games.set([])
  selectedGameId.set(0)
  metrics.set([])
  trainingState.set(null)
  leagueEntries.set([])
  leagueResults.set([])
  eloHistory.set([])
  historicalLibrary.set([])
  gauntletResults.set([])
  leagueTransitions.set([])
})
```

Add test blocks:

```js
describe('handleMessage — init with dropped data streams', () => {
  it('populates historicalLibrary from init message', () => {
    handleMessage({
      type: 'init',
      games: [], metrics: [], training_state: null,
      league_entries: [], league_results: [], elo_history: [],
      historical_library: [{ slot_index: 0, entry_name: 'Bot-A' }],
      gauntlet_results: [],
      transitions: [],
    })
    expect(get(historicalLibrary)).toHaveLength(1)
    expect(get(historicalLibrary)[0].slot_index).toBe(0)
  })

  it('populates gauntletResults from init message', () => {
    handleMessage({
      type: 'init',
      games: [], metrics: [], training_state: null,
      league_entries: [], league_results: [], elo_history: [],
      historical_library: [],
      gauntlet_results: [{ id: 1, epoch: 5, wins: 3 }],
      transitions: [],
    })
    expect(get(gauntletResults)).toHaveLength(1)
    expect(get(gauntletResults)[0].epoch).toBe(5)
  })

  it('populates leagueTransitions from msg.transitions key', () => {
    handleMessage({
      type: 'init',
      games: [], metrics: [], training_state: null,
      league_entries: [], league_results: [], elo_history: [],
      historical_library: [],
      gauntlet_results: [],
      transitions: [{ id: 1, from_role: 'dynamic', to_role: 'frontier_static' }],
    })
    // Key mapping: msg.transitions -> leagueTransitions store
    expect(get(leagueTransitions)).toHaveLength(1)
    expect(get(leagueTransitions)[0].from_role).toBe('dynamic')
  })

  it('defaults new stores to [] when keys are absent', () => {
    handleMessage({ type: 'init', games: [], metrics: [], training_state: null })
    expect(get(historicalLibrary)).toEqual([])
    expect(get(gauntletResults)).toEqual([])
    expect(get(leagueTransitions)).toEqual([])
  })
})

describe('handleMessage — league_update with dropped data streams', () => {
  it('populates all 3 new stores from league_update', () => {
    handleMessage({
      type: 'league_update',
      entries: [], results: [], elo_history: [],
      historical_library: [{ slot_index: 0 }],
      gauntlet_results: [{ id: 1 }],
      transitions: [{ id: 1 }],
    })
    expect(get(historicalLibrary)).toHaveLength(1)
    expect(get(gauntletResults)).toHaveLength(1)
    expect(get(leagueTransitions)).toHaveLength(1)
  })

  it('defaults new stores to [] when keys are absent in league_update', () => {
    handleMessage({ type: 'league_update' })
    expect(get(historicalLibrary)).toEqual([])
    expect(get(gauntletResults)).toEqual([])
    expect(get(leagueTransitions)).toEqual([])
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd webui && npx vitest run src/lib/ws.test.js`
Expected: FAIL — new stores not set in `handleMessage`

- [ ] **Step 3: Wire stores in `ws.js`**

In `webui/src/lib/ws.js`, update the import (line 10):

```js
import {
  leagueEntries, leagueResults, eloHistory, diffLeagueEntries,
  historicalLibrary, gauntletResults, leagueTransitions,
} from '../stores/league.js'
```

In the `init` case (after line 70, after `eloHistory.set(...)`), add:

```js
      historicalLibrary.set(msg.historical_library || [])
      gauntletResults.set(msg.gauntlet_results || [])
      leagueTransitions.set(msg.transitions || [])
```

In the `league_update` case (after line 125, after `eloHistory.set(...)`), add:

```js
      historicalLibrary.set(msg.historical_library || [])
      gauntletResults.set(msg.gauntlet_results || [])
      leagueTransitions.set(msg.transitions || [])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd webui && npx vitest run src/lib/ws.test.js`
Expected: All tests PASS

- [ ] **Step 5: Run all frontend tests to check for regressions**

Run: `cd webui && npx vitest run`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add webui/src/lib/ws.js webui/src/lib/ws.test.js
git commit -m "feat: wire historicalLibrary, gauntletResults, leagueTransitions in WebSocket handler"
```

---

## Task 5: Stats Banner — Replace generic metrics with role-specific Elo cards

**Files:**
- Modify: `webui/src/lib/LeagueView.svelte`

- [ ] **Step 1: Replace the entire `<script>` block**

In `LeagueView.svelte`, replace the entire `<script>` block (lines 1-14) with:

```svelte
<script>
  import { eloHistory, leagueEntries, leagueStats, learnerEntry } from '../stores/league.js'
  import { trainingState } from '../stores/training.js'
  import { buildEloChartData } from './eloChartData.js'
  import LeagueTable from './LeagueTable.svelte'
  import MatchupMatrix from './MatchupMatrix.svelte'
  import RecentMatches from './RecentMatches.svelte'
  import LeagueEventLog from './LeagueEventLog.svelte'
  import MetricsChart from './MetricsChart.svelte'

  $: chartData = buildEloChartData($eloHistory, $leagueEntries)
  $: stats = $leagueStats
  $: learner = $learnerEntry
</script>
```

- [ ] **Step 2: Replace stats banner HTML**

Replace the existing `stats-banner` div (lines 18-36) with:

```svelte
  {#if stats}
    <div class="stats-banner" role="region" aria-label="League metrics">
      <div class="stat-card highlight">
        <span class="stat-value">{learner ? Math.round(learner.elo_frontier) : '—'}</span>
        <span class="stat-label">Frontier Elo</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{learner ? Math.round(learner.elo_dynamic) : '—'}</span>
        <span class="stat-label">League Elo</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{learner ? Math.round(learner.elo_recent) : '—'}</span>
        <span class="stat-label">Challenge</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{learner ? Math.round(learner.elo_historical) : '—'}</span>
        <span class="stat-label">Gauntlet</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{stats.poolSize} / 20</span>
        <span class="stat-label">Pool</span>
      </div>
    </div>
  {/if}
```

- [ ] **Step 3: Remove unused `learnerName` prop derivation and old prop passing**

Remove `$: learnerName = $trainingState?.display_name || null` (line 13). The `LeagueTable` and `MatchupMatrix` components still receive `learnerName` as a prop — this will be addressed in Tasks 7 and 8 where those components switch to using the `learnerEntry` store directly. For now, keep passing it:

```svelte
  $: learnerName = learner?.display_name || null
```

- [ ] **Step 4: Verify the dashboard loads**

Run: `cd webui && npm run build`
Expected: Build succeeds with no errors

- [ ] **Step 5: Commit**

```bash
git add webui/src/lib/LeagueView.svelte
git commit -m "feat: replace generic stats banner with role-specific Elo metrics (W3-W6)"
```

---

## Task 6: LeagueView — Add tabbed bottom panel and EntryDetail area

**Files:**
- Modify: `webui/src/lib/LeagueView.svelte`
- Create: `webui/src/lib/EntryDetail.svelte` (stub — replaced in Task 8)
- Create: `webui/src/lib/HistoricalLibrary.svelte` (stub — replaced in Task 9)

- [ ] **Step 1: Create stub components (required — LeagueView imports them)**

```svelte
<!-- webui/src/lib/EntryDetail.svelte -->
<script>export let entryId</script>
<div class="entry-detail"><p class="empty">Entry detail for #{entryId}</p></div>
<style>.empty { color: var(--text-muted); font-size: 13px; text-align: center; padding: 24px; }</style>
```

```svelte
<!-- webui/src/lib/HistoricalLibrary.svelte -->
<div class="historical-library"><p class="empty">Historical library — not yet implemented</p></div>
<style>.empty { color: var(--text-muted); font-size: 13px; text-align: center; padding: 24px; }</style>
```

- [ ] **Step 2: Add imports for new components and store**

Add to the `<script>` block:

```js
  import HistoricalLibrary from './HistoricalLibrary.svelte'
  import EntryDetail from './EntryDetail.svelte'
  import { focusedEntryId } from '../stores/league.js'

  import { tick } from 'svelte'

  let activeBottomTab = 'recent'
  let entryDetailHeading

  function setBottomTab(tab) { activeBottomTab = tab }
  function handleTabKeydown(e) {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault()
      activeBottomTab = activeBottomTab === 'recent' ? 'history' : 'recent'
      tick().then(() => {
        document.querySelector('[role="tab"][tabindex="0"]')?.focus()
      })
    }
  }

  // Focus EntryDetail heading when an entry is selected
  $: if ($focusedEntryId != null) {
    tick().then(() => entryDetailHeading?.focus())
  }
```

- [ ] **Step 2: Replace bottom-row HTML with tabbed panel + EntryDetail**

Replace the `bottom-row` div and its contents (lines 43-48) with:

```svelte
      {#if $focusedEntryId != null}
        <div class="entry-detail-wrapper">
          <EntryDetail entryId={$focusedEntryId} bind:headingEl={entryDetailHeading} />
        </div>
      {/if}
      <div class="bottom-row">
        <div class="event-log-wrapper">
          <LeagueEventLog />
        </div>
        <div class="tabbed-panel">
          <div class="tab-bar" role="tablist" aria-label="Bottom panel">
            <button
              id="tab-recent"
              role="tab"
              aria-selected={activeBottomTab === 'recent'}
              aria-controls="panel-recent"
              tabindex={activeBottomTab === 'recent' ? 0 : -1}
              on:click={() => setBottomTab('recent')}
              on:keydown={handleTabKeydown}
              class:active={activeBottomTab === 'recent'}
            >Recent Matches</button>
            <button
              id="tab-history"
              role="tab"
              aria-selected={activeBottomTab === 'history'}
              aria-controls="panel-history"
              tabindex={activeBottomTab === 'history' ? 0 : -1}
              on:click={() => setBottomTab('history')}
              on:keydown={handleTabKeydown}
              class:active={activeBottomTab === 'history'}
            >Historical Library</button>
          </div>
          <div class="tab-content">
            {#if activeBottomTab === 'recent'}
              <div id="panel-recent" role="tabpanel" aria-labelledby="tab-recent">
                <RecentMatches />
              </div>
            {:else}
              <div id="panel-history" role="tabpanel" aria-labelledby="tab-history">
                <HistoricalLibrary />
              </div>
            {/if}
          </div>
        </div>
      </div>
```

- [ ] **Step 3: Add CSS for EntryDetail wrapper and tabbed panel**

Add to the `<style>` block:

```css
  .entry-detail-wrapper {
    max-height: 200px;
    overflow-y: auto;
    flex-shrink: 0;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-secondary);
  }

  .tabbed-panel {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }

  .tab-bar {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .tab-bar button {
    flex: 1;
    padding: 8px 12px;
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .tab-bar button.active {
    color: var(--text-primary);
    border-bottom-color: var(--accent-teal);
  }

  .tab-bar button:hover {
    color: var(--text-primary);
  }

  .tab-bar button:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: -2px;
  }

  .tab-content {
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  .tab-content > [role="tabpanel"] {
    height: 100%;
    overflow-y: auto;
  }
```

- [ ] **Step 4: Build to verify**

Run: `cd webui && npm run build`
Expected: Build succeeds (EntryDetail and HistoricalLibrary components don't exist yet — create stubs next)

- [ ] **Step 6: Commit**

```bash
git add webui/src/lib/LeagueView.svelte webui/src/lib/EntryDetail.svelte webui/src/lib/HistoricalLibrary.svelte
git commit -m "feat: add tabbed bottom panel and EntryDetail area to LeagueView"
```

---

## Task 7: LeagueTable — Add role-grouped toggle and protection badge

**Files:**
- Modify: `webui/src/lib/LeagueTable.svelte`

- [ ] **Step 1: Add grouped view state and role capacity constants**

Add to the `<script>` block of `LeagueTable.svelte`:

```js
  import { leagueByRole, focusedEntryId } from '../stores/league.js'

  const ROLE_CAPACITY = { frontier_static: 5, recent_fixed: 5, dynamic: 10, historical: 5 }
  const ROLE_ORDER = ['frontier_static', 'recent_fixed', 'dynamic', 'historical', 'other']
  const ROLE_LABELS = {
    frontier_static: '🛡 Frontier',
    recent_fixed: '✦ Recent',
    dynamic: '⚔ Dynamic',
    historical: '📜 Historical',
    other: '? Other',
  }

  let viewMode = 'flat' // 'flat' | 'grouped'
```

- [ ] **Step 2: Add toggle HTML above the table**

Insert after the `<h2>` header line, before `<div class="table-scroll">`:

```svelte
    <div class="view-toggle" role="radiogroup" aria-label="Leaderboard view">
      <button role="radio" aria-checked={viewMode === 'flat'} on:click={() => viewMode = 'flat'} class:active={viewMode === 'flat'}>Flat</button>
      <button role="radio" aria-checked={viewMode === 'grouped'} on:click={() => viewMode = 'grouped'} class:active={viewMode === 'grouped'}>Grouped</button>
    </div>
```

- [ ] **Step 3: Update `toggleExpand` to use `focusedEntryId` for below-table detail**

Replace the `toggleExpand` function and remove the inline MatchHistory expansion. The `expandedId` local variable is replaced by `focusedEntryId` store:

```js
  function toggleExpand(id) {
    focusedEntryId.update(current => current === id ? null : id)
  }
```

Remove the `let expandedId = null` declaration and replace `expandedId` references in the template with `$focusedEntryId`.

- [ ] **Step 4: Replace table body with conditional flat/grouped rendering**

Replace the `<tbody>` contents. For flat mode, keep the existing row rendering. For grouped mode, iterate by role:

```svelte
        <tbody>
          {#if viewMode === 'grouped'}
            {#each ROLE_ORDER as role}
              {#if $leagueByRole.has(role)}
                <tr class="group-header">
                  <th colspan="9" scope="colgroup">
                    <span aria-hidden="true">{ROLE_LABELS[role]?.split(' ')[0]}</span>
                    {ROLE_LABELS[role]?.split(' ').slice(1).join(' ') || role} · {$leagueByRole.get(role).length}/{ROLE_CAPACITY[role] || '?'}
                  </th>
                </tr>
                {#each $leagueByRole.get(role) as entry}
                  <tr
                    class:top={entry.rank === 1}
                    class:learner={isLearner(entry)}
                    class:focused={$focusedEntryId === entry.id}
                    aria-expanded={$focusedEntryId === entry.id}
                    on:click={() => toggleExpand(entry.id)}
                    tabindex="0"
                    on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleExpand(entry.id) }}}
                  >
                    <td class="num rank">{entry.rank}</td>
                    <td class="name-cell">
                      {entry.display_name || entry.architecture}
                      {#if isLearner(entry)}<span class="learner-badge">YOU</span>{/if}
                      {#if entry.protection_remaining > 0}
                        <span class="protection-badge" aria-label="Protected for {entry.protection_remaining} epochs"><span aria-hidden="true">🛡</span> {entry.protection_remaining}</span>
                      {/if}
                    </td>
                    <td class="num elo">{Math.round(entry.elo_rating)}</td>
                    <td class="num delta" class:delta-pos={(deltas.get(entry.id) || 0) > 0} class:delta-neg={(deltas.get(entry.id) || 0) < 0}>
                      {(deltas.get(entry.id) || 0) > 0 ? '+' : ''}{deltas.get(entry.id) || 0}
                    </td>
                    <td class="num">{entry.games_played}</td>
                    <td class="num win">{wld.get(entry.id)?.w || 0}</td>
                    <td class="num loss">{wld.get(entry.id)?.l || 0}</td>
                    <td class="num draw">{wld.get(entry.id)?.d || 0}</td>
                    <td class="num">{entry.created_epoch}</td>
                  </tr>
                {/each}
              {/if}
            {/each}
          {:else}
            {#each sorted as entry}
              <tr
                class:top={entry.rank === 1}
                class:learner={isLearner(entry)}
                class:focused={$focusedEntryId === entry.id}
                on:click={() => toggleExpand(entry.id)}
                tabindex="0"
                on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleExpand(entry.id) }}}
              >
                <td class="num rank">
                  {#if entry.rank === 1}<span class="crown" aria-hidden="true">♛</span><span class="sr-only">1 (champion)</span>{:else}{entry.rank}{/if}
                </td>
                <td class="name-cell">
                  <span class="role-badge {getRoleInfo(entry.role).cssClass}" title={getRoleInfo(entry.role).label} aria-label="{getRoleInfo(entry.role).label} tier">{getRoleInfo(entry.role).icon}</span>
                  {entry.display_name || entry.architecture}
                  {#if isLearner(entry)}<span class="learner-badge">YOU</span>{/if}
                  {#if entry.protection_remaining > 0}
                    <span class="protection-badge" aria-label="Protected for {entry.protection_remaining} epochs"><span aria-hidden="true">🛡</span> {entry.protection_remaining}</span>
                  {/if}
                </td>
                <td class="num elo">{Math.round(entry.elo_rating)}</td>
                <td class="num delta" class:delta-pos={(deltas.get(entry.id) || 0) > 0} class:delta-neg={(deltas.get(entry.id) || 0) < 0}>
                  {(deltas.get(entry.id) || 0) > 0 ? '+' : ''}{deltas.get(entry.id) || 0}
                </td>
                <td class="num">{entry.games_played}</td>
                <td class="num win">{wld.get(entry.id)?.w || 0}</td>
                <td class="num loss">{wld.get(entry.id)?.l || 0}</td>
                <td class="num draw">{wld.get(entry.id)?.d || 0}</td>
                <td class="num">{entry.created_epoch}</td>
              </tr>
            {/each}
            {#each placeholders as slot}
              <tr class="placeholder-row" aria-hidden="true">
                <td class="num rank placeholder-text">{slot}</td>
                <td class="placeholder-text">—</td>
                <td class="num placeholder-text">—</td>
                <td class="num placeholder-text"></td>
                <td class="num placeholder-text"></td>
                <td class="num placeholder-text"></td>
                <td class="num placeholder-text"></td>
                <td class="num placeholder-text"></td>
                <td class="num placeholder-text"></td>
              </tr>
            {/each}
          {/if}
        </tbody>
```

- [ ] **Step 5: Remove the old inline MatchHistory expansion rows**

Remove the `{#if expandedId === entry.id}` block and the `import MatchHistory` statement.

- [ ] **Step 6: Add CSS for toggle, group headers, protection badge, and focused state**

```css
  .view-toggle {
    display: flex;
    gap: 2px;
    margin-bottom: 8px;
  }
  .view-toggle button {
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: none;
    color: var(--text-muted);
    cursor: pointer;
  }
  .view-toggle button.active {
    background: var(--bg-card);
    color: var(--text-primary);
    border-color: var(--accent-teal);
  }
  .view-toggle button:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }

  .group-header th {
    padding: 10px 10px 6px;
    font-size: 12px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-primary);
  }

  tr.focused { background: var(--bg-card); }

  .protection-badge {
    font-size: 10px;
    color: var(--text-muted);
    margin-left: 4px;
    vertical-align: middle;
  }
```

- [ ] **Step 7: Build to verify**

Run: `cd webui && npm run build`
Expected: Build succeeds

- [ ] **Step 8: Commit**

```bash
git add webui/src/lib/LeagueTable.svelte
git commit -m "feat: add role-grouped leaderboard toggle, protection badge, wire focusedEntryId (W1, W14)"
```

---

## Task 8: Create EntryDetail component

**Files:**
- Create: `webui/src/lib/EntryDetail.svelte` (replace stub from Task 6)

- [ ] **Step 1: Implement EntryDetail component**

```svelte
<script>
  import { leagueResults, leagueEntries, headToHead } from '../stores/league.js'
  import { getRoleInfo } from './roleIcons.js'

  export let entryId
  export let headingEl = undefined

  $: entryMap = new Map($leagueEntries.map(e => [e.id, e]))
  $: entry = entryMap.get(entryId)

  // Last Round: matches from the entry's most recent epoch
  $: entryMatches = $leagueResults.filter(
    r => r.entry_a_id === entryId || r.entry_b_id === entryId
  )
  $: maxEpoch = entryMatches.length > 0
    ? Math.max(...entryMatches.map(r => r.epoch))
    : null
  $: lastRound = maxEpoch != null
    ? entryMatches.filter(r => r.epoch === maxEpoch)
    : []

  // Overall Record: aggregate by opponent from headToHead store
  $: overallOpponents = (() => {
    const opponents = []
    for (const [key, rec] of $headToHead) {
      const [aId, bId] = key.split('-').map(Number)
      if (aId === entryId) {
        const opp = entryMap.get(bId)
        if (opp) {
          opponents.push({ ...rec, opponent: opp })
        }
      }
    }
    return opponents.sort((a, b) => b.total - a.total)
  })()

  function matchPerspective(m) {
    const isA = m.entry_a_id === entryId
    const oppId = isA ? m.entry_b_id : m.entry_a_id
    const opp = entryMap.get(oppId)
    return {
      opponent: opp,
      w: isA ? (m.wins_a || 0) : (m.wins_b || 0),
      l: isA ? (m.wins_b || 0) : (m.wins_a || 0),
      d: m.draws || 0,
      eloDelta: isA
        ? Math.round((m.elo_after_a || 0) - (m.elo_before_a || 0))
        : Math.round((m.elo_after_b || 0) - (m.elo_before_b || 0)),
    }
  }
</script>

<div class="entry-detail" aria-live="polite">
  {#if !entry}
    <p class="empty">Select an entry to view details</p>
  {:else}
    <h3 class="sr-only" tabindex="-1" bind:this={headingEl}>{entry.display_name || entry.architecture} — Details</h3>

    <div class="detail-sections">
      <div class="detail-section">
        <h4 class="section-label">Last Round {#if maxEpoch != null}<span class="epoch-tag">Epoch {maxEpoch}</span>{/if}</h4>
        {#if lastRound.length === 0}
          <p class="empty-small">No matches in the current round</p>
        {:else}
          <div class="match-list">
            {#each lastRound.map(matchPerspective) as m}
              {#if m.opponent}
                <div class="match-row">
                  <span class="opp-name">
                    <span class="role-icon" aria-hidden="true">{getRoleInfo(m.opponent.role, m.opponent.status).icon}</span>
                    {m.opponent.display_name || m.opponent.architecture}
                  </span>
                  <span class="wld">{m.w}W {m.l}L {m.d}D</span>
                  <span class="elo-delta" class:positive={m.eloDelta > 0} class:negative={m.eloDelta < 0}>
                    {m.eloDelta > 0 ? '+' : ''}{m.eloDelta}
                  </span>
                </div>
              {/if}
            {/each}
          </div>
        {/if}
      </div>

      <div class="detail-section">
        <h4 class="section-label">Overall Record</h4>
        {#if overallOpponents.length === 0}
          <p class="empty-small">No match history</p>
        {:else}
          <div class="match-list">
            {#each overallOpponents as rec}
              <div class="match-row">
                <span class="opp-name">
                  <span class="role-icon" aria-hidden="true">{getRoleInfo(rec.opponent.role, rec.opponent.status).icon}</span>
                  {rec.opponent.display_name || rec.opponent.architecture}
                </span>
                <span class="wld">{rec.w}W {rec.l}L {rec.d}D</span>
                <span class="win-pct">{rec.total > 0 ? Math.round(rec.winRate * 100) : 0}%</span>
                <span class="game-count">{rec.total}g</span>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      {#if entry}
        <div class="detail-section role-stats">
          <h4 class="section-label">Role-Specific</h4>
          <div class="stat-row">
            <span class="mini-stat"><span class="mini-label">Frontier</span> {Math.round(entry.elo_frontier)}</span>
            <span class="mini-stat"><span class="mini-label">Dynamic</span> {Math.round(entry.elo_dynamic)}</span>
            <span class="mini-stat"><span class="mini-label">Recent</span> {Math.round(entry.elo_recent)}</span>
            <span class="mini-stat"><span class="mini-label">Historical</span> {Math.round(entry.elo_historical)}</span>
          </div>
          {#if entry.games_vs_frontier != null}
            <div class="stat-row games">
              <span class="mini-stat"><span class="mini-label">vs Frontier</span> {entry.games_vs_frontier}</span>
              <span class="mini-stat"><span class="mini-label">vs Dynamic</span> {entry.games_vs_dynamic}</span>
              <span class="mini-stat"><span class="mini-label">vs Recent</span> {entry.games_vs_recent}</span>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .entry-detail { padding: 10px 14px; }
  .detail-sections { display: flex; gap: 16px; flex-wrap: wrap; }
  .detail-section { flex: 1; min-width: 200px; }
  .section-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--text-muted); margin: 0 0 6px;
  }
  .epoch-tag {
    font-weight: 400; color: var(--text-muted); font-size: 10px;
    margin-left: 6px; text-transform: none; letter-spacing: 0;
  }
  .match-list { display: flex; flex-direction: column; gap: 2px; }
  .match-row {
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; padding: 2px 4px; border-radius: 3px;
  }
  .match-row:hover { background: var(--bg-card); }
  .opp-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-primary); }
  .role-icon { font-size: 10px; margin-right: 3px; }
  .wld { font-family: monospace; font-size: 11px; color: var(--text-secondary); flex-shrink: 0; }
  .elo-delta { font-family: monospace; font-size: 11px; font-weight: 600; flex-shrink: 0; min-width: 36px; text-align: right; }
  .elo-delta.positive { color: var(--accent-teal); }
  .elo-delta.negative { color: var(--danger); }
  .win-pct { font-family: monospace; font-size: 11px; color: var(--text-muted); flex-shrink: 0; }
  .game-count { font-size: 10px; color: var(--text-muted); flex-shrink: 0; }
  .role-stats { min-width: 100%; }
  .stat-row { display: flex; gap: 12px; flex-wrap: wrap; }
  .stat-row.games { margin-top: 4px; }
  .mini-stat { font-family: monospace; font-size: 12px; color: var(--text-primary); }
  .mini-label { font-size: 10px; color: var(--text-muted); margin-right: 4px; font-family: inherit; }
  .empty { color: var(--text-muted); font-size: 13px; text-align: center; padding: 24px; }
  .empty-small { color: var(--text-muted); font-size: 12px; padding: 8px 0; }
</style>
```

- [ ] **Step 2: Build to verify**

Run: `cd webui && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add webui/src/lib/EntryDetail.svelte
git commit -m "feat: create EntryDetail component with Last Round, Overall Record, Role Stats (W10)"
```

---

## Task 9: Create HistoricalLibrary component

**Files:**
- Create: `webui/src/lib/HistoricalLibrary.svelte` (replace stub from Task 6)

- [ ] **Step 1: Implement HistoricalLibrary component**

```svelte
<script>
  import { historicalLibrary, gauntletResults } from '../stores/league.js'
  import { trainingState } from '../stores/training.js'

  $: currentEpoch = $trainingState?.current_epoch || 0
  $: maxGauntletEpoch = $gauntletResults.length > 0
    ? Math.max(...$gauntletResults.map(g => g.epoch))
    : null
  $: staleness = maxGauntletEpoch != null && currentEpoch > 0
    ? currentEpoch - maxGauntletEpoch
    : null

  // Group gauntlet results by epoch (most recent first)
  $: gauntletByEpoch = (() => {
    const map = new Map()
    for (const g of $gauntletResults) {
      if (!map.has(g.epoch)) map.set(g.epoch, [])
      map.get(g.epoch).push(g)
    }
    return [...map.entries()].sort((a, b) => b[0] - a[0])
  })()
</script>

<div class="historical-library">
  <div class="slots-section">
    <h4 class="section-label">
      Library Slots
      {#if staleness != null}
        <span class="staleness">Last gauntlet: {staleness} epoch{staleness !== 1 ? 's' : ''} ago</span>
      {/if}
    </h4>
    {#if $historicalLibrary.length === 0}
      <p class="empty">No historical slots configured</p>
    {:else}
      <table>
        <caption class="sr-only">Historical library slot assignments</caption>
        <thead>
          <tr>
            <th scope="col" class="num">#</th>
            <th scope="col">Entry</th>
            <th scope="col" class="num">Target</th>
            <th scope="col" class="num">Actual</th>
            <th scope="col">Mode</th>
          </tr>
        </thead>
        <tbody>
          {#each $historicalLibrary as slot}
            <tr>
              <td class="num">{slot.slot_index}</td>
              <td>{slot.entry_name || '—'}</td>
              <td class="num">{slot.target_epoch}</td>
              <td class="num">{slot.actual_epoch ?? '—'}</td>
              <td class="mode">{slot.selection_mode}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </div>

  {#if gauntletByEpoch.length > 0}
    <div class="gauntlet-section">
      <h4 class="section-label">Gauntlet Results</h4>
      {#each gauntletByEpoch.slice(0, 5) as [epoch, results]}
        <div class="gauntlet-epoch">
          <span class="epoch-header">Epoch {epoch}</span>
          {#each results as g}
            <div class="gauntlet-row">
              <span class="slot-tag">Slot {g.historical_slot}</span>
              <span class="wld">{g.wins}W {g.losses}L {g.draws}D</span>
              {#if g.elo_before != null && g.elo_after != null}
                {@const delta = Math.round(g.elo_after - g.elo_before)}
                <span class="elo-delta" class:positive={delta > 0} class:negative={delta < 0}>
                  {delta > 0 ? '+' : ''}{delta}
                </span>
              {/if}
            </div>
          {/each}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .historical-library { padding: 10px 14px; display: flex; flex-direction: column; gap: 12px; }
  .section-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--text-muted); margin: 0 0 6px;
    display: flex; align-items: center; gap: 8px;
  }
  .staleness { font-weight: 400; font-size: 10px; color: var(--accent-gold); text-transform: none; letter-spacing: 0; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  thead { color: var(--text-muted); font-size: 11px; }
  th, td { text-align: left; padding: 3px 8px; }
  th.num, td.num { text-align: right; }
  .mode { font-size: 11px; color: var(--text-muted); }
  .gauntlet-epoch { margin-bottom: 6px; }
  .epoch-header {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--text-muted); display: block; margin-bottom: 2px;
  }
  .gauntlet-row {
    display: flex; align-items: center; gap: 8px; font-size: 12px;
    padding: 2px 4px; border-radius: 3px;
  }
  .gauntlet-row:hover { background: var(--bg-card); }
  .slot-tag { font-size: 10px; color: var(--text-muted); min-width: 48px; }
  .wld { font-family: monospace; font-size: 11px; color: var(--text-secondary); }
  .elo-delta { font-family: monospace; font-size: 11px; font-weight: 600; }
  .elo-delta.positive { color: var(--accent-teal); }
  .elo-delta.negative { color: var(--danger); }
  .empty { color: var(--text-muted); font-size: 12px; text-align: center; padding: 12px; }
</style>
```

- [ ] **Step 2: Build to verify**

Run: `cd webui && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add webui/src/lib/HistoricalLibrary.svelte
git commit -m "feat: create HistoricalLibrary component with slot table and gauntlet results (W2, W9)"
```

---

## Task 10: Event Log — Add transition count summary and batch-collapse

**Files:**
- Modify: `webui/src/lib/LeagueEventLog.svelte`

- [ ] **Step 1: Update imports and add summary display**

Replace the `<script>` block:

```svelte
<script>
  import { leagueEvents, transitionCounts } from '../stores/league.js'
  import { getRoleIcon } from './roleIcons.js'

  $: counts = $transitionCounts
</script>
```

- [ ] **Step 2: Add summary line before the event feed**

Insert after `<h2 class="section-header">Event Log</h2>`:

```svelte
  {#if counts.promotions > 0 || counts.evictions > 0 || counts.admissions > 0}
    <div class="transition-summary" aria-live="polite">
      {#if counts.promotions > 0}<span class="summary-item promotion">↑ {counts.promotions} promoted</span>{/if}
      {#if counts.evictions > 0}<span class="summary-item eviction">↓ {counts.evictions} evicted</span>{/if}
      {#if counts.admissions > 0}<span class="summary-item admission">→ {counts.admissions} admitted</span>{/if}
    </div>
  {/if}
```

- [ ] **Step 3: Add CSS for the summary line**

```css
  .transition-summary {
    display: flex;
    gap: 10px;
    font-size: 11px;
    padding: 4px 6px;
    margin-bottom: 4px;
    border-bottom: 1px solid var(--border-subtle);
  }
  .summary-item { font-weight: 600; }
  .summary-item.promotion { color: var(--accent-gold); }
  .summary-item.eviction { color: var(--danger); }
  .summary-item.admission { color: var(--accent-teal); }
```

- [ ] **Step 4: Build to verify**

Run: `cd webui && npm run build`
Expected: Build succeeds

- [ ] **Step 5: Run all frontend tests**

Run: `cd webui && npx vitest run`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add webui/src/lib/LeagueEventLog.svelte
git commit -m "feat: add transition count summary to event log (W8)"
```

---

## Task 11: Final — Run all tests and verify build

**Files:** None (verification only)

- [ ] **Step 1: Run backend tests**

Run: `uv run pytest tests/test_db.py -v`
Expected: All PASS

- [ ] **Step 2: Run frontend tests**

Run: `cd webui && npx vitest run`
Expected: All PASS

- [ ] **Step 3: Build frontend**

Run: `cd webui && npm run build`
Expected: Build succeeds

- [ ] **Step 4: Verify no lint errors**

Run: `cd webui && npx vitest run --reporter=verbose 2>&1 | head -50`
Expected: Clean output, no warnings
