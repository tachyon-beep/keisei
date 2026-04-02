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

  let container
  let chart = null
  let resizeObserver = null

  function getOpts() {
    const w = container ? container.clientWidth : width
    const colors = resolveThemeColors()
    return buildChartOpts({ width: w, height, series, compact, colors })
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

<div class="chart-wrapper">
  <div class="chart-title">{title}</div>
  <div class="chart-container" bind:this={container}></div>
  {#if annotation && !compact}
    <div class="annotation">{annotation}</div>
  {/if}
</div>

<style>
  .chart-wrapper {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
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

  .annotation {
    font-size: 16px;
    color: var(--text-muted);
    font-style: italic;
    margin-top: 6px;
    padding-left: 4px;
  }
</style>
