<script>
  import { metrics } from '../stores/metrics.js'
  import MetricsChart from './MetricsChart.svelte'
  import { extractColumns } from './metricsColumns.js'

  $: columns = extractColumns($metrics)
</script>

<div class="metrics-grid">
  <h2 class="grid-header">
    Training Metrics {#if $metrics.length > 0}— Epoch {$metrics[$metrics.length - 1]?.epoch ?? '?'}{/if}
  </h2>
  <div class="grid">
    <MetricsChart
      title="Policy & Value Loss"
      xData={columns.steps}
      series={[
        { label: 'Policy', data: columns.policyLoss, color: '#f59e0b' },
        { label: 'Value', data: columns.valueLoss, color: '#60a5fa' },
      ]}
    />
    <MetricsChart
      title="Win Rate"
      xData={columns.epochs}
      series={[
        { label: '☗ Black', data: columns.blackWinRate, color: '#e0e0e0' },
        { label: '☖ White', data: columns.whiteWinRate, color: '#60a5fa' },
        { label: 'Draw', data: columns.drawRate, color: '#f59e0b' },
      ]}
    />
    <MetricsChart
      title="Avg Episode Length"
      xData={columns.epochs}
      series={[
        { label: 'Episode Length', data: columns.avgEpLen, color: '#a78bfa' },
      ]}
      annotation="Longer games = more strategic play"
    />
    <MetricsChart
      title="Policy Entropy"
      xData={columns.steps}
      series={[
        { label: 'Entropy', data: columns.entropy, color: '#f472b6' },
      ]}
      annotation="Falling entropy = agent becoming more decisive"
    />
  </div>
</div>

<style>
  .metrics-grid {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 12px;
  }

  h2.grid-header {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
  }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }

  @media (max-width: 768px) {
    .grid {
      grid-template-columns: 1fr;
    }
  }
</style>
