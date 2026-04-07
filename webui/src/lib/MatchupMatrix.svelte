<script>
  import { leagueRanked, leagueResults, leagueEntries, focusedEntryId } from '../stores/league.js'
  import { getRoleIcon } from './roleIcons.js'

  /** Current learner display_name — used to build an aggregate "Trainer" row */
  export let learnerName = null
  /** Total grid size — pad with placeholders to fill */
  export let totalSlots = 20

  $: focused = $focusedEntryId

  // Build head-to-head data: for each (row, col) pair, aggregate W/L/D
  // Entries sorted by Elo descending (same as leaderboard)
  $: entries = $leagueRanked

  // Build aggregated trainer record across all its snapshots
  $: trainerSnapshotIds = learnerName
    ? new Set($leagueEntries.filter(e => e.display_name === learnerName).map(e => e.id))
    : new Set()

  $: hasTrainerRow = learnerName && trainerSnapshotIds.size > 0

  // Matrix participants: all entries + optionally a synthetic trainer row
  // The trainer row id is 'trainer' (string, won't collide with numeric ids)
  $: participants = (() => {
    const p = entries.map(e => ({
      id: e.id,
      label: e.display_name || e.architecture,
      shortLabel: getRoleIcon(e.role) + ' ' + shortName(e.display_name || e.architecture),
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
    while (p.length < totalSlots) {
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

  function shortName(name) {
    if (!name) return '?'
    // Truncate long names
    return name.length > 12 ? name.slice(0, 11) + '…' : name
  }

  // Build h2h map: key = "rowId-colId" => { w, l, d, winRate }
  $: h2h = (() => {
    const map = new Map()
    for (const r of $leagueResults) {
      // Direct entry-to-entry
      addResult(map, r.entry_a_id, r.entry_b_id, r.wins_a, r.wins_b, r.draws)
      // If A is a trainer snapshot, also aggregate into trainer row
      if (trainerSnapshotIds.has(r.entry_a_id)) {
        addResult(map, 'trainer', r.entry_b_id, r.wins_a, r.wins_b, r.draws)
      }
      // If B is a trainer snapshot, aggregate into trainer row (as opponent)
      if (trainerSnapshotIds.has(r.entry_b_id)) {
        addResult(map, 'trainer', r.entry_a_id, r.wins_b, r.wins_a, r.draws)
      }
    }
    // Compute win rates
    for (const [, rec] of map) {
      const total = rec.w + rec.l + rec.d
      rec.winRate = total > 0 ? rec.w / total : null
      rec.total = total
    }
    return map
  })()

  function addResult(map, rowId, colId, wins, losses, draws) {
    // row vs col
    const key = `${rowId}-${colId}`
    const rec = map.get(key) || { w: 0, l: 0, d: 0 }
    rec.w += wins || 0
    rec.l += losses || 0
    rec.d += draws || 0
    map.set(key, rec)
    // col vs row (mirror)
    const keyR = `${colId}-${rowId}`
    const recR = map.get(keyR) || { w: 0, l: 0, d: 0 }
    recR.w += losses || 0
    recR.l += wins || 0
    recR.d += draws || 0
    map.set(keyR, recR)
  }

  function cellData(rowId, colId) {
    if (rowId === colId) return null
    // Skip trainer vs its own snapshots
    if (rowId === 'trainer' && trainerSnapshotIds.has(colId)) return null
    if (colId === 'trainer' && trainerSnapshotIds.has(rowId)) return null
    return h2h.get(`${rowId}-${colId}`) || null
  }

  function cellColor(winRate) {
    if (winRate == null) return 'transparent'
    // Red (0%) → neutral (50%) → green (100%)
    if (winRate >= 0.5) {
      const t = (winRate - 0.5) * 2 // 0..1
      return `rgba(77, 184, 168, ${0.08 + t * 0.35})`
    } else {
      const t = (0.5 - winRate) * 2 // 0..1
      return `rgba(224, 80, 80, ${0.08 + t * 0.35})`
    }
  }

  function formatRate(winRate) {
    if (winRate == null) return ''
    return Math.round(winRate * 100) + '%'
  }
</script>

<div class="matrix-card">
  <h2 class="section-header">Head-to-Head</h2>

  <!-- Mobile list view -->
  <div class="h2h-list-view">
    {#if participants.filter(p => !p.isPlaceholder).length === 0}
      <p class="empty">No matchup data yet.</p>
    {:else}
      {#each participants.filter(p => !p.isPlaceholder) as row}
        {#each participants.filter(p => !p.isPlaceholder && p.id !== row.id) as col}
          {@const cell = cellData(row.id, col.id)}
          {#if cell && cell.total > 0}
            <div class="h2h-item">
              <span class="h2h-names">{row.shortLabel} vs {col.shortLabel}</span>
              <span class="h2h-record">{cell.w}W {cell.l}L {cell.d}D</span>
              <span class="h2h-rate" style="color: {cell.winRate >= 0.5 ? 'var(--accent-teal)' : 'var(--danger)'}">{formatRate(cell.winRate)}</span>
            </div>
          {/if}
        {/each}
      {/each}
    {/if}
  </div>

  <!-- Desktop matrix view -->
  <div class="matrix-desktop">
  <div class="matrix-legend" aria-label="Color legend">
    <span class="legend-swatch" style="background: rgba(224, 80, 80, 0.35)"></span>
    <span class="legend-label">0%</span>
    <span class="legend-swatch" style="background: rgba(224, 80, 80, 0.08)"></span>
    <span class="legend-swatch" style="background: transparent; border: 1px solid var(--border-subtle)"></span>
    <span class="legend-label">50%</span>
    <span class="legend-swatch" style="background: rgba(77, 184, 168, 0.08)"></span>
    <span class="legend-swatch" style="background: rgba(77, 184, 168, 0.43)"></span>
    <span class="legend-label">100%</span>
  </div>
    <div class="matrix-scroll">
      <table class="matrix" role="grid" aria-label="Head-to-head win rate matrix">
        <thead>
          <tr>
            <th class="corner"></th>
            {#each participants as col}
              <th class="col-header" class:hl={focused != null && col.id === focused} class:placeholder={col.isPlaceholder} title={col.label}>
                <span class="rotated">{col.shortLabel}</span>
              </th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each participants as row}
            <tr class:hl-row={focused != null && row.id === focused}>
              <th class="row-header" class:trainer-row={row.isTrainer} class:hl={focused != null && row.id === focused} class:placeholder={row.isPlaceholder} title={row.label}>{row.shortLabel}</th>
              {#each participants as col}
                {#if row.id === col.id}
                  <td class="self-cell" class:hl={focused != null && (row.id === focused || col.id === focused)}>—</td>
                {:else if cellData(row.id, col.id) === null}
                  <td class="no-data" class:hl={focused != null && (row.id === focused || col.id === focused)}>·</td>
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
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>
</div>

<style>
  .matrix-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    min-height: 0;
    max-height: 90%;
    overflow: hidden;
  }

  .matrix-scroll {
    overflow: auto;
    min-height: 0;
    flex: 1;
    display: flex;
    align-items: flex-start;
    justify-content: center;
  }

  .matrix {
    border-collapse: collapse;
    font-size: 15px;
    white-space: nowrap;
  }

  .corner {
    min-width: 92px;
  }

  .col-header {
    padding: 3px 6px;
    vertical-align: bottom;
  }

  .rotated {
    display: block;
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    font-size: 14px;
    font-weight: 600;
    color: var(--text-muted);
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .row-header {
    text-align: right;
    padding: 6px 10px;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-muted);
    max-width: 115px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .row-header.trainer-row {
    color: var(--accent-teal);
    font-weight: 700;
  }

  td {
    text-align: center;
    padding: 6px 8px;
    min-width: 50px;
    font-family: monospace;
    font-size: 14px;
    font-weight: 600;
    border: 1px solid var(--border-subtle);
    color: var(--text-primary);
  }

  .self-cell {
    background: var(--bg-primary);
    color: var(--text-muted);
  }

  .no-data {
    color: var(--text-muted);
    font-size: 12px;
  }

  .rate-cell {
    cursor: help;
    transition: opacity 0.1s;
  }

  .rate-cell:hover {
    opacity: 0.8;
  }

  /* Faint crosshatch highlight for focused entry's row + column */
  .hl {
    background-image: repeating-linear-gradient(
      45deg,
      transparent,
      transparent 3px,
      rgba(77, 184, 168, 0.08) 3px,
      rgba(77, 184, 168, 0.08) 4px
    );
  }

  th.hl {
    color: var(--accent-teal);
  }

  .placeholder {
    color: var(--text-muted);
    opacity: 0.35;
  }

  .empty {
    color: var(--text-muted);
    font-size: 13px;
    text-align: center;
    padding: 24px;
  }

  /* Mobile list view: shown below 768px, hidden on desktop */
  .h2h-list-view {
    display: none;
    flex-direction: column;
    gap: 2px;
    overflow-y: auto;
    flex: 1;
    min-height: 0;
  }

  .h2h-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 6px;
    font-size: 12px;
    border-radius: 3px;
  }

  .h2h-item:hover { background: var(--bg-card); }

  .h2h-names {
    flex: 1;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .h2h-record {
    font-family: monospace;
    font-size: 12px;
    color: var(--text-secondary);
    flex-shrink: 0;
  }

  .h2h-rate {
    font-family: monospace;
    font-size: 12px;
    font-weight: 600;
    flex-shrink: 0;
    min-width: 36px;
    text-align: right;
  }

  .matrix-desktop { display: contents; }

  @media (max-width: 768px) {
    .h2h-list-view { display: flex; }
    .matrix-desktop { display: none; }
  }

  @media (prefers-reduced-motion: reduce) {
    .rate-cell { transition: none; }
  }

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
    font-size: 12px;
    color: var(--text-muted);
    font-family: monospace;
  }

  .rate-cell:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: -2px;
  }
</style>
