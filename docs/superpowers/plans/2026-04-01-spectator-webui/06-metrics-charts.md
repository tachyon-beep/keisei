# Plan 6: Training Metrics Charts

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the MetricsChart component wrapping uPlot, and render the 4-chart 2x2 grid with pedagogical annotations.

**Architecture:** A reusable `MetricsChart.svelte` wraps uPlot with reactive data updates. Four instances are created for: policy/value loss, win rate, avg episode length, and policy entropy. Each chart has dark theme styling and optional annotation text.

**Tech Stack:** uPlot, Svelte

---

### Task 1: MetricsChart Component

**Files:**
- Create: `webui/src/lib/MetricsChart.svelte`

- [ ] **Step 1: Implement MetricsChart.svelte**

`webui/src/lib/MetricsChart.svelte`:
```svelte
<script>
  import { onMount, onDestroy, afterUpdate } from 'svelte'
  import uPlot from 'uplot'
  import 'uplot/dist/uPlot.min.css'

  /** @type {string} Chart title */
  export let title = ''

  /** @type {string|null} Pedagogical annotation shown below the chart */
  export let annotation = null

  /** @type {Array<number>} X-axis data (e.g., steps or episodes) */
  export let xData = []

  /** @type {Array<{label: string, data: Array<number>, color: string}>} Series configs */
  export let series = []

  /** @type {number} Chart width */
  export let width = 400

  /** @type {number} Chart height */
  export let height = 120

  let container
  let chart = null

  const darkTheme = {
    background: '#0d1117',
    gridColor: '#222',
    textColor: '#888',
    axisColor: '#555',
  }

  function buildOpts() {
    return {
      width,
      height,
      padding: [8, 8, 0, 0],
      cursor: { show: true },
      legend: { show: true },
      axes: [
        {
          stroke: darkTheme.textColor,
          grid: { stroke: darkTheme.gridColor, width: 0.5 },
          ticks: { stroke: darkTheme.axisColor },
          font: '10px sans-serif',
        },
        {
          stroke: darkTheme.textColor,
          grid: { stroke: darkTheme.gridColor, width: 0.5 },
          ticks: { stroke: darkTheme.axisColor },
          font: '10px sans-serif',
        },
      ],
      series: [
        { label: 'X' },
        ...series.map(s => ({
          label: s.label,
          stroke: s.color,
          width: 1.5,
          fill: s.color + '20',
        })),
      ],
    }
  }

  function buildData() {
    return [
      xData,
      ...series.map(s => s.data),
    ]
  }

  onMount(() => {
    if (container && xData.length > 0) {
      chart = new uPlot(buildOpts(), buildData(), container)
    }
  })

  afterUpdate(() => {
    if (!container) return
    if (xData.length === 0) return

    if (chart) {
      chart.setData(buildData())
    } else {
      chart = new uPlot(buildOpts(), buildData(), container)
    }
  })

  onDestroy(() => {
    if (chart) {
      chart.destroy()
      chart = null
    }
  })
</script>

<div class="chart-wrapper">
  <div class="chart-title">{title}</div>
  <div class="chart-container" bind:this={container}></div>
  {#if annotation}
    <div class="annotation">{annotation}</div>
  {/if}
</div>

<style>
  .chart-wrapper {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
  }

  .chart-title {
    font-size: 10px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
  }

  .chart-container {
    width: 100%;
  }

  /* Override uPlot legend styles for dark theme */
  .chart-container :global(.u-legend) {
    font-size: 10px !important;
    color: var(--text-secondary) !important;
  }

  .annotation {
    font-size: 11px;
    color: var(--text-muted);
    font-style: italic;
    margin-top: 6px;
    padding-left: 4px;
  }
</style>
```

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/MetricsChart.svelte
git commit -m "feat: reusable uPlot chart component with dark theme"
```

---

### Task 2: Metrics Grid Component

**Files:**
- Create: `webui/src/lib/MetricsGrid.svelte`

- [ ] **Step 1: Implement MetricsGrid.svelte**

This component subscribes to the metrics store and renders four charts in a 2x2 grid.

`webui/src/lib/MetricsGrid.svelte`:
```svelte
<script>
  import { metrics } from '../stores/metrics.js'
  import MetricsChart from './MetricsChart.svelte'

  // Extract series data from metrics rows
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
```

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/MetricsGrid.svelte
git commit -m "feat: 2x2 metrics chart grid with pedagogical annotations"
```
