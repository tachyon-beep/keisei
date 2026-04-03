<script>
  import { onMount, onDestroy, afterUpdate } from 'svelte'
  import uPlot from 'uplot'
  import 'uplot/dist/uPlot.min.css'
  import { buildChartOpts, buildChartData, resolveThemeColors } from './chartHelpers.js'
  import { theme } from '../stores/theme.js'

  export let title = ''
  export let annotation = null
  export let xData = []
  export let series = []
  export let width = 400
  export let height = 100
  export let compact = false
  export let xLabel = null
  /** 'bottom' (default uPlot) or 'right' (custom side legend) */
  export let legendPosition = 'bottom'

  let container
  let chart = null
  let resizeObserver = null

  $: sideLegend = legendPosition === 'right'

  function getOpts() {
    const w = container ? container.clientWidth : width
    const colors = resolveThemeColors()
    const opts = buildChartOpts({ width: w, height, series, compact, xLabel, colors })
    if (sideLegend) {
      opts.legend = { show: false }
    }
    return opts
  }

  function getData() {
    return buildChartData(xData, series)
  }

  function rebuildChart() {
    if (!container || xData.length === 0) return
    if (chart) { chart.destroy(); chart = null }
    chart = new uPlot(getOpts(), getData(), container)
  }

  // Rebuild chart when theme changes so axis colors update
  const unsubTheme = theme.subscribe(() => {
    // Skip initial subscription (before mount)
    if (container) {
      // Allow DOM to apply new data-theme attribute first
      requestAnimationFrame(rebuildChart)
    }
  })

  onMount(() => {
    if (container && xData.length > 0) {
      chart = new uPlot(getOpts(), getData(), container)
    }

    resizeObserver = new ResizeObserver(entries => {
      if (chart && entries[0]) {
        const { width: w } = entries[0].contentRect
        if (w > 0) {
          chart.setSize({ width: Math.floor(w), height })
        }
      }
    })
    if (container) resizeObserver.observe(container)
  })

  afterUpdate(() => {
    if (!container) return
    if (xData.length === 0) return

    if (chart) {
      chart.setData(getData())
    } else {
      chart = new uPlot(getOpts(), getData(), container)
    }
  })

  onDestroy(() => {
    unsubTheme()
    if (resizeObserver) {
      resizeObserver.disconnect()
      resizeObserver = null
    }
    if (chart) {
      chart.destroy()
      chart = null
    }
  })
</script>

<div class="chart-wrapper" class:has-side-legend={sideLegend}>
  <div class="chart-body">
    <div class="chart-title">{title}</div>
    <div class="chart-container" bind:this={container}></div>
    {#if annotation && !compact}
      <div class="annotation">{annotation}</div>
    {/if}
  </div>
  {#if sideLegend && series.length > 0}
    <div class="side-legend" role="list" aria-label="Chart legend">
      {#each series as s}
        <div class="legend-item" role="listitem">
          <span class="legend-swatch" style="background: {s.color}"></span>
          <span class="legend-label">{s.label}</span>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .chart-wrapper {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
  }

  .chart-wrapper.has-side-legend {
    display: flex;
    gap: 12px;
    align-items: stretch;
  }

  .chart-body {
    flex: 1;
    min-width: 0;
  }

  .chart-title {
    font-size: 12px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
  }

  .chart-container {
    width: 100%;
  }

  .chart-container :global(.u-legend) {
    font-size: 12px !important;
    color: var(--text-secondary) !important;
  }

  .side-legend {
    flex: 0 0 auto;
    max-width: 180px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 4px 0;
    border-left: 1px solid var(--border-subtle);
    padding-left: 12px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-secondary);
    line-height: 1.3;
  }

  .legend-swatch {
    flex-shrink: 0;
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }

  .legend-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .annotation {
    font-size: 16px;
    color: var(--text-muted);
    font-style: italic;
    margin-top: 6px;
    padding-left: 4px;
  }
</style>
