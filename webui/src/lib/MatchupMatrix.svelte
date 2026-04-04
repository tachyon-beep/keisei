<script>
  import { leagueRanked, leagueResults, leagueEntries, focusedEntryId } from '../stores/league.js'

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
    // Pad to totalSlots with placeholders
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
      addResult(map, r.learner_id, r.opponent_id, r.wins, r.losses, r.draws)
      // If learner is a trainer snapshot, also aggregate into trainer row
      if (trainerSnapshotIds.has(r.learner_id)) {
        addResult(map, 'trainer', r.opponent_id, r.wins, r.losses, r.draws)
      }
      // If opponent is a trainer snapshot, aggregate into trainer row (as opponent)
      if (trainerSnapshotIds.has(r.opponent_id)) {
        addResult(map, 'trainer', r.learner_id, r.losses, r.wins, r.draws)
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
    font-size: 10px;
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

  @media (prefers-reduced-motion: reduce) {
    .rate-cell { transition: none; }
  }
</style>
