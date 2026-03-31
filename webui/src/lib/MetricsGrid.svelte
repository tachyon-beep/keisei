<script>
  import { metrics } from '../stores/metrics.js'
  import MetricsChart from './MetricsChart.svelte'

  $: steps = $metrics.map(r => r.step || 0)
  $: policyLoss = $metrics.map(r => r.policy_loss ?? null)
  $: valueLoss = $metrics.map(r => r.value_loss ?? null)
  $: winRate = $metrics.map(r => r.win_rate ?? null)
  $: avgEpLen = $metrics.map(r => r.avg_episode_length ?? null)
  $: entropy = $metrics.map(r => r.entropy ?? null)
  $: epochs = $metrics.map(r => r.epoch || 0)
</script>

<div class="metrics-grid">
  <div class="grid-header">
    Training Metrics {#if $metrics.length > 0}— Epoch {$metrics[$metrics.length - 1]?.epoch ?? '?'}{/if}
  </div>
  <div class="grid">
    <MetricsChart
      title="Policy & Value Loss"
      xData={steps}
      series={[
        { label: 'Policy', data: policyLoss, color: '#f59e0b' },
        { label: 'Value', data: valueLoss, color: '#60a5fa' },
      ]}
    />
    <MetricsChart
      title="Win Rate"
      xData={epochs}
      series={[
        { label: 'Win Rate', data: winRate, color: '#4ade80' },
      ]}
    />
    <MetricsChart
      title="Avg Episode Length"
      xData={epochs}
      series={[
        { label: 'Episode Length', data: avgEpLen, color: '#a78bfa' },
      ]}
      annotation="Longer games = more strategic play"
    />
    <MetricsChart
      title="Policy Entropy"
      xData={steps}
      series={[
        { label: 'Entropy', data: entropy, color: '#f472b6' },
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

  .grid-header {
    font-size: 10px;
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
</style>
