<script>
  import { eloHistory, leagueEntries, leagueResults, leagueStats, learnerEntry, historicalLibrary, tournamentStats, displayElo } from '../stores/league.js'
  import { trainingState } from '../stores/training.js'
  import { buildEloChartData } from './eloChartData.js'
  import LeagueTable from './LeagueTable.svelte'
  import MatchupMatrix from './MatchupMatrix.svelte'
  import RecentMatches from './RecentMatches.svelte'
  import LeagueEventLog from './LeagueEventLog.svelte'
  import MetricsChart from './MetricsChart.svelte'
  import HistoricalLibrary from './HistoricalLibrary.svelte'
  import EntryDetail from './EntryDetail.svelte'
  import { focusedEntryId } from '../stores/league.js'
  import { tick } from 'svelte'

  $: chartData = buildEloChartData($eloHistory, $leagueEntries)
  $: stats = $leagueStats
  $: tStats = $tournamentStats
  $: learner = $learnerEntry
  $: learnerName = learner?.display_name || null

  // Active pool = Frontier(5) + Recent(5) + Dynamic(10). Historical(5) are library entries, not pool members.
  const POOL_CAPACITY = 20

  let activeBottomTab = 'recent'
  let entryDetailHeading

  function closeDetail() { focusedEntryId.set(null) }
  function handleDetailKeydown(e) {
    if (e.key === 'Escape') { e.preventDefault(); closeDetail() }
  }
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

<main id="league-main" class="league-view" aria-label="League standings">
  {#if stats}
    <div class="stats-banner" role="region" aria-label="League summary">
      <div class="stat-card">
        <span class="stat-value">{stats.poolSize} / {POOL_CAPACITY}</span>
        <span class="stat-label">Pool Size</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{stats.totalMatches}</span>
        <span class="stat-label">Matches</span>
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

  <div class="league-grid">
    <div class="left-top">
      <div class="table-wrapper">
        <LeagueTable {learnerName} />
      </div>
    </div>
    <div class="right-top">
      {#if $focusedEntryId != null}
        <div class="entry-detail-wrapper" role="region" aria-label="Entry detail" on:keydown={handleDetailKeydown}>
          <button class="detail-close-btn" on:click={closeDetail} aria-label="Close entry detail">✕</button>
          <EntryDetail entryId={$focusedEntryId} bind:headingEl={entryDetailHeading} />
        </div>
      {:else}
        <MatchupMatrix {learnerName} />
      {/if}
    </div>
    <div class="left-bottom">
      <div class="event-log-wrapper">
        <LeagueEventLog />
      </div>
      <div class="tabbed-panel">
        <nav aria-label="Bottom panel"><div class="tab-bar" role="tablist">
          <button
            id="tab-recent"
            role="tab"
            aria-selected={activeBottomTab === 'recent'}
            aria-controls="panel-recent"
            tabindex={activeBottomTab === 'recent' ? 0 : -1}
            on:click={() => setBottomTab('recent')}
            on:keydown={handleTabKeydown}
            class:active={activeBottomTab === 'recent'}
          >Recent Matches{#if $leagueResults.length > 0} <span class="tab-count">({$leagueResults.length})</span>{/if}</button>
          <button
            id="tab-history"
            role="tab"
            aria-selected={activeBottomTab === 'history'}
            aria-controls="panel-history"
            tabindex={activeBottomTab === 'history' ? 0 : -1}
            on:click={() => setBottomTab('history')}
            on:keydown={handleTabKeydown}
            class:active={activeBottomTab === 'history'}
          >Historical Library{#if $historicalLibrary.length > 0} <span class="tab-count">({$historicalLibrary.length})</span>{/if}</button>
        </div></nav>
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
    <div class="chart-card">
      <h2 class="section-header">Elo Over Time</h2>
      {#if chartData.xData.length > 0}
        <MetricsChart
          title=""
          xData={chartData.xData}
          series={chartData.series}
          height={200}
          xLabel="Epoch"
          legendPosition="right"
        />
      {:else}
        <p class="empty">No matches yet.</p>
      {/if}
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

  .league-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 3fr 1fr;
    gap: 12px;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  .left-top {
    min-height: 0;
    overflow: hidden;
  }

  .table-wrapper {
    height: 100%;
    min-height: 0;
    overflow: hidden;
  }

  .right-top {
    min-height: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .left-bottom {
    display: flex;
    gap: 12px;
    min-height: 0;
    overflow: hidden;
  }

  .left-bottom > :global(*) {
    flex: 1;
    min-width: 0;
    min-height: 0;
  }

  .event-log-wrapper {
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .chart-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .chart-card > :global(.chart-wrapper) {
    flex: 1;
    min-height: 0;
  }

  .empty {
    color: var(--text-muted);
    font-size: 13px;
    text-align: center;
    padding: 24px;
  }

  .entry-detail-wrapper {
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-secondary);
    position: relative;
    flex: 1;
    min-height: 0;
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
    min-width: 22px;
    min-height: 22px;
    padding: 2px 4px;
    font-size: 11px;
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
    padding: 10px 12px;
    min-height: 44px;
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

  .tab-count {
    font-weight: 400;
    opacity: 0.7;
    font-size: 11px;
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

  @media (max-width: 1200px) {
    .league-grid {
      grid-template-columns: 1fr;
      grid-template-rows: auto auto auto auto;
      overflow-y: auto;
    }

    .left-bottom {
      flex-direction: column;
    }
  }
</style>
