<script>
  import { leagueRanked, entryWLD, eloDelta, focusedEntryId } from '../stores/league.js'
  import MatchHistory from './MatchHistory.svelte'

  /** Current learner's display_name (used to highlight their row) */
  export let learnerName = null
  /** Total slots shown in leaderboard (empty ones are placeholders) */
  export let totalSlots = 20

  let sortColumn = 'elo_rating'
  let sortAsc = false
  let expandedId = null

  $: wld = $entryWLD
  $: deltas = $eloDelta

  $: sorted = (() => {
    const entries = [...$leagueRanked]
    entries.sort((a, b) => {
      let av, bv
      if (sortColumn === 'wins') {
        av = (wld.get(a.id)?.w || 0); bv = (wld.get(b.id)?.w || 0)
      } else if (sortColumn === 'losses') {
        av = (wld.get(a.id)?.l || 0); bv = (wld.get(b.id)?.l || 0)
      } else if (sortColumn === 'draws') {
        av = (wld.get(a.id)?.d || 0); bv = (wld.get(b.id)?.d || 0)
      } else if (sortColumn === 'elo_delta') {
        av = (deltas.get(a.id) || 0); bv = (deltas.get(b.id) || 0)
      } else {
        av = a[sortColumn]; bv = b[sortColumn]
      }
      return sortAsc ? (av > bv ? 1 : -1) : (bv > av ? 1 : -1)
    })
    return entries.map((e, i) => ({ ...e, rank: i + 1 }))
  })()

  function toggleSort(col) {
    if (sortColumn === col) {
      sortAsc = !sortAsc
    } else {
      sortColumn = col
      sortAsc = false
    }
  }

  function toggleExpand(id) {
    expandedId = expandedId === id ? null : id
    focusedEntryId.set(expandedId)
  }

  function sortIndicator(col) {
    if (sortColumn !== col) return ''
    return sortAsc ? ' ▲' : ' ▼'
  }

  function ariaSortValue(col) {
    if (sortColumn !== col) return 'none'
    return sortAsc ? 'ascending' : 'descending'
  }

  function isLearner(entry) {
    return learnerName && entry.display_name === learnerName
  }

  $: placeholderCount = Math.max(0, totalSlots - sorted.length)
  $: placeholders = Array.from({ length: placeholderCount }, (_, i) => sorted.length + i + 1)
</script>

<div class="league-table-card">
  <h2 class="section-header">Elo Leaderboard {#if placeholderCount > 0}<span class="slot-count">{sorted.length} / {totalSlots}</span>{/if}</h2>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th class="num" aria-sort="none">#</th>
            <th aria-sort={ariaSortValue('display_name')}>
              <button class="sort-btn" on:click={() => toggleSort('display_name')}>Name{sortIndicator('display_name')}</button>
            </th>
            <th class="num" aria-sort={ariaSortValue('elo_rating')}>
              <button class="sort-btn" on:click={() => toggleSort('elo_rating')}>Elo{sortIndicator('elo_rating')}</button>
            </th>
            <th class="num" aria-sort={ariaSortValue('elo_delta')}>
              <button class="sort-btn" on:click={() => toggleSort('elo_delta')}>±{sortIndicator('elo_delta')}</button>
            </th>
            <th class="num" aria-sort={ariaSortValue('games_played')}>
              <button class="sort-btn" on:click={() => toggleSort('games_played')}>GP{sortIndicator('games_played')}</button>
            </th>
            <th class="num wld-col" aria-sort={ariaSortValue('wins')}>
              <button class="sort-btn" on:click={() => toggleSort('wins')}>W{sortIndicator('wins')}</button>
            </th>
            <th class="num wld-col" aria-sort={ariaSortValue('losses')}>
              <button class="sort-btn" on:click={() => toggleSort('losses')}>L{sortIndicator('losses')}</button>
            </th>
            <th class="num wld-col" aria-sort={ariaSortValue('draws')}>
              <button class="sort-btn" on:click={() => toggleSort('draws')}>D{sortIndicator('draws')}</button>
            </th>
            <th class="num" aria-sort={ariaSortValue('created_epoch')}>
              <button class="sort-btn" on:click={() => toggleSort('created_epoch')}>Epoch{sortIndicator('created_epoch')}</button>
            </th>
          </tr>
        </thead>
        <tbody>
          {#each sorted as entry}
            <tr
              class:top={entry.rank === 1}
              class:learner={isLearner(entry)}
              class:expanded={expandedId === entry.id}
              on:click={() => toggleExpand(entry.id)}
              aria-expanded={expandedId === entry.id}
              tabindex="0"
              on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleExpand(entry.id) }}}
            >
              <td class="num rank">
                {#if entry.rank === 1}
                  <span class="crown" aria-label="Rank 1">♛</span>
                {:else}
                  {entry.rank}
                {/if}
              </td>
              <td class="name-cell">
                {entry.display_name || entry.architecture}
                {#if isLearner(entry)}
                  <span class="learner-badge">YOU</span>
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
            {#if expandedId === entry.id}
              <tr class="history-row">
                <td colspan="9">
                  <MatchHistory entryId={entry.id} />
                </td>
              </tr>
            {/if}
          {/each}
          {#each placeholders as slot}
            <tr class="placeholder-row">
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
        </tbody>
      </table>
    </div>
</div>

<style>
  .league-table-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .table-scroll {
    flex: 1;
    overflow-y: scroll;
    min-height: 0;
  }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead { color: var(--text-muted); font-size: 13px; position: sticky; top: 0; background: var(--bg-secondary); z-index: 1; }
  th, td { text-align: left; padding: 6px 10px; }
  th.num, td.num { text-align: right; }
  th.wld-col { min-width: 36px; }

  .sort-btn {
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    color: var(--text-muted);
    font-size: 13px;
    font-weight: 600;
  }
  .sort-btn:hover { color: var(--text-primary); }
  .sort-btn:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }

  tbody tr {
    border-bottom: 1px solid var(--border-subtle);
    cursor: pointer;
    color: var(--text-primary);
    transition: background 0.1s;
  }
  tbody tr:hover { background: var(--bg-card); }
  tbody tr:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: -2px; }

  /* Rank #1 row accent */
  tr.top {
    border-left: 3px solid var(--accent-gold);
  }
  tr.top .rank { color: var(--accent-gold); font-weight: 700; }
  tr.top .elo { color: var(--accent-teal); font-weight: 700; }

  .crown {
    color: var(--accent-gold);
    font-size: 15px;
  }

  /* Current learner highlight */
  tr.learner {
    background: rgba(77, 184, 168, 0.08);
    border-left: 3px solid var(--accent-teal);
  }
  tr.learner:hover {
    background: rgba(77, 184, 168, 0.14);
  }

  .learner-badge {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: var(--accent-teal);
    background: rgba(77, 184, 168, 0.12);
    padding: 1px 5px;
    border-radius: 3px;
    margin-left: 6px;
    vertical-align: middle;
  }

  .name-cell {
    white-space: nowrap;
  }

  tr.expanded { background: var(--bg-card); }

  .history-row { cursor: default; }
  .history-row:hover { background: transparent; }
  .history-row td { padding: 0; }

  .elo { font-family: monospace; }
  .delta { font-family: monospace; font-size: 12px; color: var(--text-muted); }
  .delta-pos { color: var(--accent-teal); }
  .delta-neg { color: var(--danger); }
  .win { color: var(--accent-teal); font-family: monospace; }
  .loss { color: var(--danger); font-family: monospace; }
  .draw { color: var(--accent-gold); font-family: monospace; }

  .slot-count {
    font-weight: 400;
    color: var(--text-muted);
    font-size: 11px;
    margin-left: 6px;
  }

  .placeholder-row {
    cursor: default;
  }
  .placeholder-row:hover { background: transparent; }
  .placeholder-text {
    color: var(--text-muted);
    opacity: 0.35;
    font-size: 12px;
    user-select: none;
  }

  .empty { color: var(--text-muted); font-size: 13px; padding: 24px; text-align: center; }

  @media (prefers-reduced-motion: reduce) {
    tbody tr { transition: none; }
  }
</style>
