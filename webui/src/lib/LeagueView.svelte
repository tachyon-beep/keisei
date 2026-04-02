<script>
  import { eloHistory, leagueEntries } from '../stores/league.js'
  import { buildEloChartData } from './eloChartData.js'
  import LeagueTable from './LeagueTable.svelte'
  import MetricsChart from './MetricsChart.svelte'

  $: chartData = buildEloChartData($eloHistory, $leagueEntries)
</script>

<div class="league-view">
  <LeagueTable />

  <div class="elo-chart-section">
    <h2 class="section-header">Elo Over Time</h2>
    {#if chartData.xData.length > 0}
      <MetricsChart
        title=""
        xData={chartData.xData}
        series={chartData.series}
        height={250}
      />
    {:else}
      <p class="empty">Elo history will appear after league matches are played.</p>
    {/if}
  </div>
</div>

<style>
  .league-view {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .elo-chart-section {
    padding: 0 12px 12px;
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
  }
</style>
