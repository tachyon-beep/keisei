<script>
  import { onMount, onDestroy, afterUpdate } from 'svelte'
  import uPlot from 'uplot'
  import 'uplot/dist/uPlot.min.css'

  export let title = ''
  export let annotation = null
  export let xData = []
  export let series = []
  export let width = 400
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
