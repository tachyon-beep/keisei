<script>
  import { eloHistory, leagueEntries, leagueStats, learnerEntry } from '../stores/league.js'
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
  $: learner = $learnerEntry
  $: learnerName = learner?.display_name || null

  // Active pool = Frontier(5) + Recent(5) + Dynamic(10). Historical(5) are library entries, not pool members.
  const POOL_CAPACITY = 20

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

<main class="league-view" aria-label="League standings">
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
        <span class="stat-label">Challenge Score</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{learner ? Math.round(learner.elo_historical) : '—'}</span>
        <span class="stat-label">Gauntlet Score</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{stats?.poolSize ?? '—'} / {POOL_CAPACITY}</span>
        <span class="stat-label">Pool</span>
      </div>
    </div>

  <div class="league-columns">
    <div class="left-column">
      <div class="table-wrapper">
        <LeagueTable {learnerName} />
      </div>
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
    </div>
    <div class="right-column">
      <MatchupMatrix {learnerName} />
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

  .league-columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  .left-column {
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 12px;
    overflow: hidden;
  }

  .table-wrapper {
    flex: 0 1 auto;
    min-height: 0;
    overflow: hidden;
    max-height: 65%;
  }

  .chart-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .chart-card > :global(.chart-wrapper) {
    flex: 1;
    min-height: 0;
  }

  .bottom-row {
    display: flex;
    gap: 12px;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  .bottom-row > :global(*) {
    flex: 1;
    min-width: 0;
    min-height: 0;
  }

  .event-log-wrapper {
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .right-column {
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 12px;
    overflow: hidden;
  }

  .empty {
    color: var(--text-muted);
    font-size: 13px;
    text-align: center;
    padding: 24px;
  }

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
</style>
