<script>
  import { onMount, onDestroy } from 'svelte'
  import { winProbHistory, showcaseSelectedPly, showcaseDisplayedMove } from '../stores/showcase.js'
  import uPlot from 'uplot'

  let chartEl
  let containerEl
  let chart = null
  let resizeObserver = null

  function buildData(history) {
    if (!history || history.length === 0) return [[], []]
    return [
      history.map(h => h.ply),
      history.map(h => h.value),
    ]
  }

  // uPlot plugin: draws a vertical marker at the displayed (scrubbed) ply so
  // spectators can see where in the trajectory they're paused.
  let markerPly = null
  function drawMarker(u) {
    if (markerPly == null) return
    const ctx = u.ctx
    const x = u.valToPos(markerPly, 'x', true)
    if (!isFinite(x)) return
    ctx.save()
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent-gold').trim() || '#c8962e'
    ctx.lineWidth = 2
    ctx.setLineDash([3, 3])
    ctx.beginPath()
    ctx.moveTo(x, u.bbox.top)
    ctx.lineTo(x, u.bbox.top + u.bbox.height)
    ctx.stroke()
    ctx.restore()
  }

  function makeOpts(width) {
    return {
      width,
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
        { stroke: 'var(--accent-teal)', width: 2, fill: 'rgba(77, 184, 168, 0.12)' },
      ],
      hooks: { draw: [drawMarker] },
    }
  }

  function currentWidth() {
    if (!containerEl) return 300
    const w = containerEl.clientWidth
    return Math.max(120, w)
  }

  onMount(() => {
    chart = new uPlot(makeOpts(currentWidth()), buildData($winProbHistory), chartEl)
    if (typeof ResizeObserver !== 'undefined' && containerEl) {
      resizeObserver = new ResizeObserver(() => {
        if (chart) chart.setSize({ width: currentWidth(), height: 120 })
      })
      resizeObserver.observe(containerEl)
    }
  })

  onDestroy(() => {
    if (resizeObserver) resizeObserver.disconnect()
    if (chart) chart.destroy()
  })

  $: if (chart && $winProbHistory) {
    chart.setData(buildData($winProbHistory))
  }

  // Repaint marker when the displayed ply changes (scrubbing or live tick).
  $: if (chart && $showcaseDisplayedMove) {
    markerPly = $showcaseDisplayedMove.ply
    chart.redraw(false, true)
  } else if (chart && !$showcaseDisplayedMove) {
    markerPly = null
    chart.redraw(false, true)
  }
</script>

<div class="win-prob-graph">
  <h3 class="section-label">Win Probability</h3>
  <div class="chart-host" bind:this={containerEl}>
    <div bind:this={chartEl}></div>
  </div>
</div>

<style>
  .win-prob-graph { padding: 8px; }
  .section-label { font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin: 0 0 4px; }
  .chart-host { width: 100%; min-width: 0; }
</style>
