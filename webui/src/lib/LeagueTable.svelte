<script>
  import { leagueRanked } from '../stores/league.js'
  import MatchHistory from './MatchHistory.svelte'

  let sortColumn = 'elo_rating'
  let sortAsc = false
  let expandedId = null

  $: sorted = (() => {
    const entries = [...$leagueRanked]
    entries.sort((a, b) => {
      const av = a[sortColumn], bv = b[sortColumn]
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
  }

  function sortIndicator(col) {
    if (sortColumn !== col) return ''
    return sortAsc ? ' ▲' : ' ▼'
  }
</script>

<div class="league-table">
  <h2 class="section-header">Elo Leaderboard</h2>
  {#if sorted.length === 0}
    <p class="empty">No league entries yet. League data appears once opponent pool training begins.</p>
  {:else}
    <table>
      <thead>
        <tr>
          <th class="num">#</th>
          <th><button class="sort-btn" on:click={() => toggleSort('architecture')}>Model{sortIndicator('architecture')}</button></th>
          <th class="num"><button class="sort-btn" on:click={() => toggleSort('elo_rating')}>Elo{sortIndicator('elo_rating')}</button></th>
          <th class="num"><button class="sort-btn" on:click={() => toggleSort('games_played')}>Games{sortIndicator('games_played')}</button></th>
          <th class="num"><button class="sort-btn" on:click={() => toggleSort('created_epoch')}>Epoch{sortIndicator('created_epoch')}</button></th>
        </tr>
      </thead>
      <tbody>
        {#each sorted as entry}
          <tr
            class:top={entry.rank === 1}
            class:expanded={expandedId === entry.id}
            on:click={() => toggleExpand(entry.id)}
            aria-expanded={expandedId === entry.id}
            tabindex="0"
            on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleExpand(entry.id) }}}
          >
            <td class="num rank">{entry.rank}</td>
            <td>{entry.architecture}</td>
            <td class="num elo">{Math.round(entry.elo_rating)}</td>
            <td class="num">{entry.games_played}</td>
            <td class="num">{entry.created_epoch}</td>
          </tr>
          {#if expandedId === entry.id}
            <tr class="history-row">
              <td colspan="5">
                <MatchHistory entryId={entry.id} />
              </td>
            </tr>
          {/if}
        {/each}
      </tbody>
    </table>
  {/if}
</div>

<style>
  .league-table { padding: 12px; }

  .section-header {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
  }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead { color: var(--text-muted); font-size: 13px; }
  th, td { text-align: left; padding: 6px 10px; }
  th.num, td.num { text-align: right; }

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
  tbody tr:hover { background: var(--bg-secondary); }
  tbody tr:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: -2px; }

  tr.top .rank { color: var(--accent-gold); font-weight: 700; }
  tr.top .elo { color: var(--accent-teal); font-weight: 700; }
  tr.expanded { background: var(--bg-secondary); }

  .history-row { cursor: default; }
  .history-row:hover { background: transparent; }
  .history-row td { padding: 0; }

  .elo { font-family: monospace; }
  .empty { color: var(--text-muted); font-size: 13px; padding: 24px; text-align: center; }

  @media (prefers-reduced-motion: reduce) {
    tbody tr { transition: none; }
  }
</style>
