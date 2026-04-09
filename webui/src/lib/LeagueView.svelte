<script>
  import { leagueEntries, leagueResults, leagueStats, learnerEntry, tournamentStats, displayElo } from '../stores/league.js'
  import { trainingState } from '../stores/training.js'
  import LeagueTable from './LeagueTable.svelte'
  import MatchupMatrix from './MatchupMatrix.svelte'
  import RecentMatches from './RecentMatches.svelte'
  import LeagueEventLog from './LeagueEventLog.svelte'
  import EntryDetail from './EntryDetail.svelte'
  import { focusedEntryId } from '../stores/league.js'
  import { tick } from 'svelte'

  $: stats = $leagueStats
  $: tStats = $tournamentStats
  $: learner = $learnerEntry
  $: learnerName = learner?.display_name || null

  // Active pool = Frontier(5) + Recent(5) + Dynamic(10). Historical(5) are library entries, not pool members.
  const POOL_CAPACITY = 20

  let entryDetailHeading

  function closeDetail() { focusedEntryId.set(null) }
  function handleMainKeydown(e) {
    if (e.key === 'Escape' && $focusedEntryId != null) {
      e.preventDefault()
      closeDetail()
    }
  }

  // Focus EntryDetail heading only when focusedEntryId actually changes value
  let prevFocusedId = null
  $: {
    const currentId = $focusedEntryId
    if (currentId !== prevFocusedId) {
      prevFocusedId = currentId
      if (currentId != null) {
        tick().then(() => entryDetailHeading?.focus())
      }
    }
  }
</script>

<main id="league-main" class="league-view" aria-label="League standings" on:keydown={handleMainKeydown}>
  {#if stats}
    <div class="stats-banner" role="region" aria-label="League summary">
      <div class="stat-card">
        <span class="stat-value">{stats.poolSize} / {POOL_CAPACITY}</span>
        <span class="stat-label">Pool Size</span>
      </div>
      <div class="stat-card stat-trio">
        <div class="trio-item">
          <span class="stat-value">{stats.totalRounds}</span>
          <span class="stat-label">Rounds</span>
        </div>
        <span class="trio-sep" aria-hidden="true"></span>
        <div class="trio-item">
          <span class="stat-value">{stats.totalMatches}</span>
          <span class="stat-label">Matches</span>
        </div>
        <span class="trio-sep" aria-hidden="true"></span>
        <div class="trio-item">
          <span class="stat-value">{stats.totalGames}</span>
          <span class="stat-label">Games</span>
        </div>
      </div>
      <div class="stat-card highlight">
        <span class="stat-value">{stats.topEntry?.display_name || stats.topEntry?.architecture || '—'}</span>
        <span class="stat-label">Top Rated · {Math.round(stats.topEntry ? displayElo(stats.topEntry).value : 0)}</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{stats.eloMin} – {stats.eloMax}</span>
        <span class="stat-label">Elo Range · {stats.eloSpread} spread</span>
      </div>
      {#if tStats}
        <div class="stat-card">
          <span class="stat-value">{Math.round(tStats.games_per_min)}</span>
          <span class="stat-label">Games/min · {tStats.active_slots} slots</span>
        </div>
      {/if}
    </div>
  {/if}

  <div class="league-grid" class:has-detail={$focusedEntryId != null}>
    <div class="left-col">
      <div class="table-wrapper">
        <LeagueTable />
      </div>
      {#if $focusedEntryId != null}
        <div class="entry-detail-wrapper" role="region" aria-label="Entry detail">
          <button class="detail-close-btn" on:click={closeDetail} aria-label="Close entry detail">✕</button>
          <EntryDetail entryId={$focusedEntryId} bind:headingEl={entryDetailHeading} />
        </div>
      {/if}
    </div>
    <div class="right-col">
      <MatchupMatrix {learnerName} />
    </div>
    <div class="bottom-left">
      <div class="event-log-wrapper">
        <LeagueEventLog />
      </div>
    </div>
    <div class="bottom-right">
      <div class="recent-matches-wrapper">
        <RecentMatches />
      </div>
    </div>
  </div>
</main>

<style>
  .league-view {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 12px 16px;
    height: 100%;
    min-height: 0;
    overflow: hidden;
  }

  .stats-banner {
    display: flex;
    gap: 12px;
    flex-shrink: 0;
  }

  .stat-card {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 14px 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
  }

  .stat-card.highlight {
    border-color: var(--accent-gold);
    background: rgba(200, 150, 46, 0.06);
  }

  .stat-value {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
    font-family: monospace;
  }

  .stat-card.highlight .stat-value {
    color: var(--accent-gold);
  }

  .stat-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .stat-trio {
    flex-direction: row;
    justify-content: center;
    gap: 0;
    padding: 10px 8px;
  }

  .trio-item {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .trio-sep {
    width: 1px;
    align-self: stretch;
    margin: 4px 0;
    background: var(--border);
  }

  /* --- Main grid: 2 columns, 2 rows --- */
  .league-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 3fr 1fr;
    gap: 12px;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  /* Left column: table + optional detail */
  .left-col {
    grid-column: 1;
    grid-row: 1;
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 0;
    overflow: hidden;
  }

  .table-wrapper {
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  /* When detail is open, table shrinks to share space */
  .has-detail .table-wrapper {
    flex: 1;
    min-height: 180px;
  }

  .entry-detail-wrapper {
    flex: 0 1 auto;
    min-height: 120px;
    max-height: 60%;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-secondary);
    position: relative;
  }

  .detail-close-btn {
    position: absolute;
    top: 4px;
    right: 4px;
    z-index: 1;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-secondary);
    cursor: pointer;
    min-width: 44px;
    min-height: 44px;
    padding: 2px 4px;
    font-size: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .detail-close-btn:hover {
    color: var(--text-primary);
    border-color: var(--text-secondary);
    background: var(--bg-card);
  }

  .detail-close-btn:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }

  /* Right column: matchup matrix, always visible */
  .right-col {
    grid-column: 2;
    grid-row: 1;
    min-height: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  /* Bottom cells: align with columns above */
  .bottom-left,
  .bottom-right {
    min-height: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .recent-matches-wrapper {
    height: 100%;
    min-height: 0;
    overflow: hidden;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
  }

  .event-log-wrapper {
    height: 100%;
    min-height: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  @media (max-width: 1200px) {
    .league-grid {
      grid-template-columns: 1fr;
      grid-template-rows: auto auto auto auto;
      overflow-y: auto;
    }

    .bottom-left,
    .bottom-right {
      max-height: none;
    }
  }
</style>
