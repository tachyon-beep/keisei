<script>
  import { eloHistory, leagueEntries, leagueStats } from '../stores/league.js'
  import { trainingState } from '../stores/training.js'
  import { buildEloChartData } from './eloChartData.js'
  import LeagueTable from './LeagueTable.svelte'
  import MetricsChart from './MetricsChart.svelte'

  $: chartData = buildEloChartData($eloHistory, $leagueEntries)
  $: stats = $leagueStats
  $: learnerName = $trainingState?.display_name || null
</script>

<main class="league-view" aria-label="League standings">
  {#if stats}
    <div class="stats-banner" role="region" aria-label="League summary">
      <div class="stat-card">
        <span class="stat-value">{stats.poolSize}</span>
        <span class="stat-label">Pool Size</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{stats.totalMatches}</span>
        <span class="stat-label">Matches</span>
      </div>
      <div class="stat-card highlight">
        <span class="stat-value">{stats.topEntry?.display_name || stats.topEntry?.architecture || '—'}</span>
        <span class="stat-label">Top Rated · {Math.round(stats.topEntry?.elo_rating || 0)}</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{stats.eloMin} – {stats.eloMax}</span>
        <span class="stat-label">Elo Range · {stats.eloSpread} spread</span>
      </div>
    </div>
  {/if}

  <div class="league-columns">
    <div class="table-column">
      <LeagueTable {learnerName} />
    </div>
    <div class="chart-column">
      <div class="chart-card">
        <h2 class="section-header">Elo Over Time</h2>
        {#if chartData.xData.length > 0}
          <MetricsChart
            title=""
            xData={chartData.xData}
            series={chartData.series}
            height={400}
          />
        {:else}
          <p class="empty">Elo history will appear after league matches are played.</p>
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
    overflow: auto;
  }

  .stats-banner {
    display: flex;
    gap: 12px;
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
    font-size: 11px;
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
  }

  .table-column {
    min-height: 0;
    overflow: auto;
  }

  .chart-column {
    min-height: 0;
    display: flex;
    flex-direction: column;
  }

  .chart-card {
    flex: 1;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    display: flex;
    flex-direction: column;
  }

  .section-header {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
  }

  .empty {
    color: var(--text-muted);
    font-size: 13px;
    text-align: center;
    padding: 24px;
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
  }
</style>
