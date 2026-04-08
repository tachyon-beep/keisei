<script>
  import { onMount, onDestroy } from 'svelte'
  import { winProbHistory } from '../stores/showcase.js'
  import uPlot from 'uplot'

  let chartEl
  let chart = null

  function buildData(history) {
    if (!history || history.length === 0) return [[], []]
    return [
      history.map(h => h.ply),
      history.map(h => h.value),
    ]
  }

  const opts = {
    width: 300,
    height: 120,
    cursor: { show: false },
    legend: { show: false },
    scales: { y: { range: [0, 1] } },
    axes: [
      { show: true, size: 20, font: '10px sans-serif', stroke: 'var(--text-muted)' },
      { show: true, size: 30, font: '10px sans-serif', stroke: 'var(--text-muted)',
        values: (u, vals) => vals.map(v => (v * 100).toFixed(0) + '%') },
    ],
    series: [
      {},
      { stroke: 'var(--accent-teal)', width: 2, fill: 'rgba(0, 180, 180, 0.1)' },
    ],
  }

  onMount(() => {
    chart = new uPlot(opts, buildData($winProbHistory), chartEl)
  })

  onDestroy(() => {
    if (chart) chart.destroy()
  })

  $: if (chart && $winProbHistory) {
    chart.setData(buildData($winProbHistory))
  }
</script>

<div class="win-prob-graph">
  <h3 class="section-label">Win Probability</h3>
  <div bind:this={chartEl}></div>
</div>

<style>
  .win-prob-graph { padding: 8px; }
  .section-label { font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin: 0 0 4px; }
</style>
