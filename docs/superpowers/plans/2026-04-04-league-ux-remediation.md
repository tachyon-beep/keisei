# League Page UX Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all critical, major, and minor UX issues identified in the league page design review — accessibility, text sizing, interaction affordances, responsive layout, and code deduplication.

**Architecture:** All changes are scoped to `webui/src/`. Shared CSS goes into `app.css`. Component changes are isolated per-file. No new dependencies. No store changes needed — all fixes are presentational or ARIA-related.

**Tech Stack:** Svelte 4, CSS custom properties, Vitest (for existing test suites — no new test files needed since changes are CSS/markup)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `webui/src/app.css` | Modify | Add shared `.section-header` class, add responsive breakpoint for league |
| `webui/src/lib/LeagueView.svelte` | Modify | Remove local `.section-header`, add responsive stacking, stat-label sizing |
| `webui/src/lib/LeagueTable.svelte` | Modify | Column tooltips, crown a11y, sort affordance, placeholder aria-hidden, caption, scroll fix |
| `webui/src/lib/LeagueEventLog.svelte` | Modify | Text sizing, icon a11y, remove local `.section-header`, scroll fix |
| `webui/src/lib/RecentMatches.svelte` | Modify | Text sizing, empty state text, remove local `.section-header`, scroll fix |
| `webui/src/lib/MatchupMatrix.svelte` | Modify | Dynamic slot count, cell keyboard access, color legend, remove local `.section-header` |
| `webui/src/lib/MatchHistory.svelte` | Modify | Add table caption |
| `webui/src/index.html` | Modify | Remove Noto Serif font import |
| `webui/src/lib/StatusIndicator.svelte` | Modify | Replace Noto Serif with system serif fallback |

---

### Task 1: Extract shared `.section-header` to `app.css`

**Files:**
- Modify: `webui/src/app.css` (add at end, before closing)
- Modify: `webui/src/lib/LeagueView.svelte:187-194` (remove local style)
- Modify: `webui/src/lib/LeagueTable.svelte:173-181` (remove local style)
- Modify: `webui/src/lib/LeagueEventLog.svelte:36-43` (remove local style)
- Modify: `webui/src/lib/RecentMatches.svelte:132-139` (remove local style)
- Modify: `webui/src/lib/MatchupMatrix.svelte:180-187` (remove local style)

- [ ] **Step 1: Add `.section-header` to `app.css`**

Add at end of `webui/src/app.css`:

```css
/* ── Shared component styles ─────────────────── */
.section-header {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 10px;
  flex-shrink: 0;
}
```

- [ ] **Step 2: Remove the local `.section-header` from each component**

In each of these five files, delete the scoped `.section-header { ... }` block from the `<style>` section. The files and line ranges:

- `LeagueView.svelte`: lines 187-194
- `LeagueTable.svelte`: lines 173-181
- `LeagueEventLog.svelte`: lines 36-43 (note: this one has `margin-bottom: 8px` instead of 10px — the global 10px is fine)
- `RecentMatches.svelte`: lines 132-139
- `MatchupMatrix.svelte`: lines 180-187

- [ ] **Step 3: Visual check**

Run: `cd /home/john/keisei/webui && npm run dev`

Open the League tab. Verify all five section headers ("Elo Leaderboard", "Event Log", "Recent Matches", "Head-to-Head", "Elo Over Time") render identically — 12px uppercase, `--text-secondary` color, 1px letter-spacing.

- [ ] **Step 4: Commit**

```bash
git add webui/src/app.css webui/src/lib/LeagueView.svelte webui/src/lib/LeagueTable.svelte webui/src/lib/LeagueEventLog.svelte webui/src/lib/RecentMatches.svelte webui/src/lib/MatchupMatrix.svelte
git commit -m "refactor(webui): extract shared .section-header to app.css

Removes 5 identical scoped copies across league components."
```

---

### Task 2: Fix minimum text sizes (10-11px to 12px)

**Files:**
- Modify: `webui/src/lib/LeagueView.svelte:119` (stat-label)
- Modify: `webui/src/lib/LeagueEventLog.svelte:59,72` (event body + timestamp)
- Modify: `webui/src/lib/RecentMatches.svelte:154,218` (epoch-separator + match-detail)

- [ ] **Step 1: Fix stat-label in LeagueView.svelte**

In `LeagueView.svelte`, in the `<style>` block, change `.stat-label`:

```css
  .stat-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
```

Changed: `font-size: 11px` → `font-size: 12px`.

- [ ] **Step 2: Fix event log text sizes in LeagueEventLog.svelte**

Change `.event` font-size from 11px to 12px:

```css
  .event {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 6px;
    font-size: 12px;
    border-radius: 3px;
  }
```

Change `.event-time` font-size from 10px to 11px (monospace at 11px reads equivalent to 12px proportional):

```css
  .event-time {
    color: var(--text-muted);
    font-family: monospace;
    font-size: 11px;
    flex-shrink: 0;
    min-width: 60px;
  }
```

- [ ] **Step 3: Fix match detail text sizes in RecentMatches.svelte**

Change `.epoch-separator` font-size from 10px to 11px:

```css
  .epoch-separator {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    padding: 6px 8px 3px;
    border-bottom: 1px solid var(--border-subtle);
    margin-top: 4px;
  }
```

Change `.match-detail` font-size from 10px to 11px:

```css
  .match-detail {
    font-size: 11px;
    color: var(--text-muted);
    display: flex;
    gap: 4px;
    padding-left: 2px;
  }
```

- [ ] **Step 4: Visual check**

Open League tab. Verify:
- Stats banner labels are readable at the new size
- Event log entries are slightly larger — no layout overflow
- Recent Matches detail lines (game count, win%, clash#) are legible

- [ ] **Step 5: Commit**

```bash
git add webui/src/lib/LeagueView.svelte webui/src/lib/LeagueEventLog.svelte webui/src/lib/RecentMatches.svelte
git commit -m "fix(webui): bump minimum text sizes from 10-11px to 11-12px

Improves readability of stat labels, event log, and match details."
```

---

### Task 3: Fix crown/rank screen reader parity (Critical)

**Files:**
- Modify: `webui/src/lib/LeagueTable.svelte:113-118`

- [ ] **Step 1: Replace crown-only rank cell with crown + visually-hidden number**

In `LeagueTable.svelte`, find the rank cell template (lines 113-118):

```svelte
              <td class="num rank">
                {#if entry.rank === 1}
                  <span class="crown" aria-hidden="true">♛</span>
                  <span class="sr-only">1 (champion)</span>
                {:else}
                  {entry.rank}
                {/if}
              </td>
```

Changes: `aria-label` on the crown → `aria-hidden="true"` on the crown. Added a `sr-only` span with the actual rank number and "champion" context.

- [ ] **Step 2: Add `.sr-only` utility to `app.css`**

Add to `webui/src/app.css` (after the `.section-header` block):

```css
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
```

- [ ] **Step 3: Commit**

```bash
git add webui/src/lib/LeagueTable.svelte webui/src/app.css
git commit -m "fix(a11y): crown emoji now has visually-hidden rank number for screen readers

Critical: rank 1 previously replaced the number entirely with a crown
emoji, losing positional context for AT users."
```

---

### Task 4: Add tooltips to abbreviated column headers

**Files:**
- Modify: `webui/src/lib/LeagueTable.svelte:76-99`

- [ ] **Step 1: Add `title` attributes to sort buttons**

Replace the `<thead>` block in `LeagueTable.svelte` (lines 73-100):

```svelte
        <thead>
          <tr>
            <th class="num" aria-sort="none">#</th>
            <th aria-sort={ariaSortValue('display_name')}>
              <button class="sort-btn" on:click={() => toggleSort('display_name')} title="Sort by name">Name{sortIndicator('display_name')}</button>
            </th>
            <th class="num" aria-sort={ariaSortValue('elo_rating')}>
              <button class="sort-btn" on:click={() => toggleSort('elo_rating')} title="Sort by Elo rating">Elo{sortIndicator('elo_rating')}</button>
            </th>
            <th class="num" aria-sort={ariaSortValue('elo_delta')}>
              <button class="sort-btn" on:click={() => toggleSort('elo_delta')} title="Sort by Elo change">±{sortIndicator('elo_delta')}</button>
            </th>
            <th class="num" aria-sort={ariaSortValue('games_played')}>
              <button class="sort-btn" on:click={() => toggleSort('games_played')} title="Sort by games played">GP{sortIndicator('games_played')}</button>
            </th>
            <th class="num wld-col" aria-sort={ariaSortValue('wins')}>
              <button class="sort-btn" on:click={() => toggleSort('wins')} title="Sort by wins">W{sortIndicator('wins')}</button>
            </th>
            <th class="num wld-col" aria-sort={ariaSortValue('losses')}>
              <button class="sort-btn" on:click={() => toggleSort('losses')} title="Sort by losses">L{sortIndicator('losses')}</button>
            </th>
            <th class="num wld-col" aria-sort={ariaSortValue('draws')}>
              <button class="sort-btn" on:click={() => toggleSort('draws')} title="Sort by draws">D{sortIndicator('draws')}</button>
            </th>
            <th class="num" aria-sort={ariaSortValue('created_epoch')}>
              <button class="sort-btn" on:click={() => toggleSort('created_epoch')} title="Sort by creation epoch">Epoch{sortIndicator('created_epoch')}</button>
            </th>
          </tr>
        </thead>
```

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/LeagueTable.svelte
git commit -m "fix(a11y): add title tooltips to abbreviated leaderboard column headers"
```

---

### Task 5: Make sort affordance visible on all sortable columns

**Files:**
- Modify: `webui/src/lib/LeagueTable.svelte` (style block, lines 195-205)

- [ ] **Step 1: Add hover underline and subtle icon hint to sort buttons**

Replace the `.sort-btn` styles in `LeagueTable.svelte`:

```css
  .sort-btn {
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    color: var(--text-muted);
    font-size: 13px;
    font-weight: 600;
    text-decoration: none;
  }
  .sort-btn:hover {
    color: var(--text-primary);
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  .sort-btn:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
```

The underline-on-hover signals clickability on columns that don't yet show ▲/▼.

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/LeagueTable.svelte
git commit -m "fix(ux): add hover underline to sort buttons for discoverability"
```

---

### Task 6: Fix event log icon accessibility

**Files:**
- Modify: `webui/src/lib/LeagueEventLog.svelte:12-16`

- [ ] **Step 1: Wrap icons in `aria-hidden` and add visually-hidden labels**

Replace the event template (lines 12-16) in `LeagueEventLog.svelte`:

```svelte
        <div class="event" class:arrival={event.type === 'arrival'} class:departure={event.type === 'departure'} class:promotion={event.type === 'promotion'} class:demotion={event.type === 'demotion'}>
          <span class="event-time">{event.time}</span>
          <span class="event-icon" aria-hidden="true">{event.icon}</span>
          <span class="sr-only">{event.type}</span>
          <span class="event-name">{event.name}</span>
          <span class="event-detail">{event.detail}</span>
        </div>
```

Changes: Added `aria-hidden="true"` to the icon span. Added `<span class="sr-only">{event.type}</span>` so screen readers announce "arrival", "departure", "promotion", or "demotion" instead of "rightwards arrow".

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/LeagueEventLog.svelte
git commit -m "fix(a11y): hide arrow icons from AT, add visually-hidden event type labels"
```

---

### Task 7: Reduce matchup matrix placeholder count

**Files:**
- Modify: `webui/src/lib/MatchupMatrix.svelte:24-51`

- [ ] **Step 1: Change placeholder padding from `totalSlots` to `entries.length + 3`**

In `MatchupMatrix.svelte`, replace the `participants` reactive block (lines 24-51):

```svelte
  $: participants = (() => {
    const p = entries.map(e => ({
      id: e.id,
      label: e.display_name || e.architecture,
      shortLabel: shortName(e.display_name || e.architecture),
      isTrainer: false,
    }))
    if (hasTrainerRow) {
      p.unshift({
        id: 'trainer',
        label: learnerName + ' (all)',
        shortLabel: shortName(learnerName) + '*',
        isTrainer: true,
      })
    }
    // Show a few placeholder slots to hint at pool capacity, not the full 20
    const targetSize = Math.min(p.length + 3, totalSlots)
    while (p.length < targetSize) {
      const slot = p.length + 1
      p.push({
        id: `empty-${slot}`,
        label: `Slot ${slot}`,
        shortLabel: `#${slot}`,
        isTrainer: false,
        isPlaceholder: true,
      })
    }
    return p
  })()
```

Change: replaced `while (p.length < totalSlots)` with `while (p.length < targetSize)` where `targetSize = Math.min(p.length + 3, totalSlots)`. This shows at most 3 placeholder slots instead of padding to 20.

- [ ] **Step 2: Visual check**

Open League tab with a small pool (e.g. 3 entries). Verify the matrix shows ~6-7 rows/columns (3 entries + trainer row + 3 placeholders) instead of a sparse 20x20 grid.

- [ ] **Step 3: Commit**

```bash
git add webui/src/lib/MatchupMatrix.svelte
git commit -m "fix(ux): cap matchup matrix placeholders at +3 instead of padding to 20

Avoids a sparse 20x20 grid when only a few entries are in the pool."
```

---

### Task 8: Add keyboard access and color legend to matchup matrix

**Files:**
- Modify: `webui/src/lib/MatchupMatrix.svelte:126-165` (template), `webui/src/lib/MatchupMatrix.svelte:167-302` (styles)

- [ ] **Step 1: Add color legend below the header**

In `MatchupMatrix.svelte`, after the `<h2 class="section-header">Head-to-Head</h2>` line (line 127), add:

```svelte
  <div class="matrix-legend" aria-label="Color legend">
    <span class="legend-swatch" style="background: rgba(224, 80, 80, 0.35)"></span>
    <span class="legend-label">0%</span>
    <span class="legend-swatch" style="background: rgba(224, 80, 80, 0.08)"></span>
    <span class="legend-swatch" style="background: transparent; border: 1px solid var(--border-subtle)"></span>
    <span class="legend-swatch" style="background: rgba(77, 184, 168, 0.08)"></span>
    <span class="legend-swatch" style="background: rgba(77, 184, 168, 0.43)"></span>
    <span class="legend-label">100%</span>
  </div>
```

- [ ] **Step 2: Add `tabindex` and `aria-label` to rate cells**

Replace the rate-cell `<td>` (around line 150-157):

```svelte
                {:else}
                  <td
                    class="rate-cell"
                    class:hl={focused != null && (row.id === focused || col.id === focused)}
                    style="background: {cellColor(cellData(row.id, col.id).winRate)}"
                    title="{row.label} vs {col.label}: {cellData(row.id, col.id).w}W {cellData(row.id, col.id).l}L {cellData(row.id, col.id).d}D ({cellData(row.id, col.id).total} games)"
                    tabindex="0"
                    aria-label="{row.label} vs {col.label}: {formatRate(cellData(row.id, col.id).winRate)} win rate, {cellData(row.id, col.id).w} wins, {cellData(row.id, col.id).l} losses, {cellData(row.id, col.id).d} draws"
                  >
                    {formatRate(cellData(row.id, col.id).winRate)}
                  </td>
                {/if}
```

- [ ] **Step 3: Add legend styles**

Add to the `<style>` block in `MatchupMatrix.svelte`:

```css
  .matrix-legend {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-bottom: 8px;
    flex-shrink: 0;
  }

  .legend-swatch {
    width: 16px;
    height: 10px;
    border-radius: 2px;
    display: inline-block;
  }

  .legend-label {
    font-size: 10px;
    color: var(--text-muted);
    font-family: monospace;
  }

  .rate-cell:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: -2px;
  }
```

- [ ] **Step 4: Visual check**

Open League tab. Verify:
- Color legend strip appears between "Head-to-Head" header and the matrix grid
- Tab key can navigate between rate cells
- Focus ring is visible on focused cells

- [ ] **Step 5: Commit**

```bash
git add webui/src/lib/MatchupMatrix.svelte
git commit -m "fix(a11y): add color legend and keyboard access to matchup matrix

Adds win-rate color gradient legend. Rate cells now have tabindex and
aria-label for screen reader and keyboard users."
```

---

### Task 9: Add `<caption>` to data tables

**Files:**
- Modify: `webui/src/lib/LeagueTable.svelte:72`
- Modify: `webui/src/lib/MatchHistory.svelte:23`

- [ ] **Step 1: Add caption to leaderboard table**

In `LeagueTable.svelte`, after `<table>` (line 72), add:

```svelte
      <table>
        <caption class="sr-only">Elo leaderboard, sorted by {sortColumn} {sortAsc ? 'ascending' : 'descending'}</caption>
```

- [ ] **Step 2: Add caption to match history table**

In `MatchHistory.svelte`, after `<table>` (line 23), add:

```svelte
    <table>
      <caption class="sr-only">Match history for selected entry</caption>
```

- [ ] **Step 3: Commit**

```bash
git add webui/src/lib/LeagueTable.svelte webui/src/lib/MatchHistory.svelte
git commit -m "fix(a11y): add visually-hidden captions to leaderboard and match history tables"
```

---

### Task 10: Add `aria-hidden` to placeholder rows

**Files:**
- Modify: `webui/src/lib/LeagueTable.svelte:144`

- [ ] **Step 1: Add `aria-hidden="true"` to placeholder rows**

In `LeagueTable.svelte`, change the placeholder `<tr>` (line 144):

```svelte
          {#each placeholders as slot}
            <tr class="placeholder-row" aria-hidden="true">
```

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/LeagueTable.svelte
git commit -m "fix(a11y): hide placeholder rows from screen readers"
```

---

### Task 11: Differentiate empty states

**Files:**
- Modify: `webui/src/lib/LeagueEventLog.svelte:8`
- Modify: `webui/src/lib/RecentMatches.svelte:78`

- [ ] **Step 1: Change empty state messages**

In `LeagueEventLog.svelte`, change line 8:

```svelte
    <p class="empty">No league events yet.</p>
```

In `RecentMatches.svelte`, change line 78:

```svelte
    <p class="empty">No matches played yet.</p>
```

These are now distinct from each other and from a loading state (which would say "Connecting...").

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/LeagueEventLog.svelte webui/src/lib/RecentMatches.svelte
git commit -m "fix(ux): differentiate empty state messages in event log and recent matches"
```

---

### Task 12: Switch `overflow-y: scroll` to `auto`

**Files:**
- Modify: `webui/src/lib/LeagueTable.svelte:185`
- Modify: `webui/src/lib/LeagueEventLog.svelte:47`
- Modify: `webui/src/lib/RecentMatches.svelte:143`

- [ ] **Step 1: Replace `scroll` with `auto` in all three**

In `LeagueTable.svelte`, change `.table-scroll`:
```css
    overflow-y: auto;
```

In `LeagueEventLog.svelte`, change `.feed`:
```css
    overflow-y: auto;
```

In `RecentMatches.svelte`, change `.feed`:
```css
    overflow-y: auto;
```

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/LeagueTable.svelte webui/src/lib/LeagueEventLog.svelte webui/src/lib/RecentMatches.svelte
git commit -m "fix(ux): use overflow-y auto instead of scroll to hide unnecessary scrollbars"
```

---

### Task 13: Add responsive breakpoint for narrow viewports

**Files:**
- Modify: `webui/src/lib/LeagueView.svelte` (style block)

- [ ] **Step 1: Add media query to stack columns below 1200px**

Add at the end of the `<style>` block in `LeagueView.svelte`, before the closing `</style>`:

```css
  @media (max-width: 1200px) {
    .league-columns {
      grid-template-columns: 1fr;
      overflow-y: auto;
    }

    .table-wrapper {
      max-height: none;
    }

    .bottom-row {
      flex-direction: column;
    }
  }
```

- [ ] **Step 2: Visual check**

Resize browser to < 1200px width. Verify:
- Left column stacks above right column
- Leaderboard table is no longer height-constrained
- Event log and recent matches stack vertically instead of side-by-side

- [ ] **Step 3: Commit**

```bash
git add webui/src/lib/LeagueView.svelte
git commit -m "fix(ux): add responsive breakpoint to stack league columns below 1200px"
```

---

### Task 14: Remove unused Noto Serif font import

**Files:**
- Modify: `webui/index.html:6-9`
- Modify: `webui/src/lib/StatusIndicator.svelte:141`

- [ ] **Step 1: Remove Google Fonts lines from `index.html`**

In `webui/index.html`, remove lines 6-9 (the `preconnect` and font link):

```html
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Noto+Serif:wght@400;600;700&display=swap" rel="stylesheet" />
```

The result should be:

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="description" content="Real-time training dashboard for Keisei Shogi DRL system" />
    <title>Keisei Training Dashboard</title>
  </head>
```

- [ ] **Step 2: Replace Noto Serif in StatusIndicator.svelte**

In `StatusIndicator.svelte`, change line 141:

```css
    font-family: Georgia, 'Times New Roman', serif;
```

This preserves the serif aesthetic without the external font load.

- [ ] **Step 3: Commit**

```bash
git add webui/index.html webui/src/lib/StatusIndicator.svelte
git commit -m "perf(webui): remove Noto Serif Google Font import (~30KB savings)

Only StatusIndicator used it; replaced with system serif fallback."
```

---

### Task 15: Final visual and accessibility check

- [ ] **Step 1: Run existing tests**

```bash
cd /home/john/keisei/webui && npm test
```

Expected: all existing tests pass (changes are CSS/markup only, no logic changes).

- [ ] **Step 2: Full visual review**

Open League tab and verify:
- All section headers render consistently
- Text is readable at new sizes
- Crown shows visually, rank 1 is accessible
- Column headers show tooltips on hover
- Sort buttons show underline on hover
- Matchup matrix has color legend, cells are tabbable
- Placeholder grid is compact (not 20x20)
- Scrollbars only appear when needed
- At < 1200px, layout stacks vertically
- No Noto Serif network request in devtools Network tab

- [ ] **Step 3: Keyboard navigation check**

Tab through the league page:
1. Sort buttons in leaderboard header are reachable
2. Leaderboard rows are reachable, Enter/Space expands
3. Matchup matrix cells are reachable, focus ring visible
4. No tab traps

- [ ] **Step 4: Final commit if any touch-ups needed**

Only if manual review found issues. Otherwise, done.
