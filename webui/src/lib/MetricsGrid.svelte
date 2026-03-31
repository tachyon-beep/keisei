<script>
  import { metrics } from '../stores/metrics.js'
  import MetricsChart from './MetricsChart.svelte'

  $: columns = (() => {
    const steps = [], policyLoss = [], valueLoss = [], winRate = [], avgEpLen = [], entropy = [], epochs = []
    for (const r of $metrics) {
      steps.push(r.step || 0)
      policyLoss.push(r.policy_loss ?? null)
      valueLoss.push(r.value_loss ?? null)
      winRate.push(r.win_rate ?? null)
      avgEpLen.push(r.avg_episode_length ?? null)
      entropy.push(r.entropy ?? null)
      epochs.push(r.epoch || 0)
    }
    return { steps, policyLoss, valueLoss, winRate, avgEpLen, entropy, epochs }
  })()
</script>

<div class="metrics-grid">
  <div class="grid-header">
    Training Metrics {#if $metrics.length > 0}— Epoch {$metrics[$metrics.length - 1]?.epoch ?? '?'}{/if}
  </div>
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
        { label: 'Win Rate', data: columns.winRate, color: '#4ade80' },
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
