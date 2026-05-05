<script>
  import { onMount, onDestroy } from 'svelte'
  import { winProbHistory, showcaseSelectedPly, showcaseDisplayedMove, showcaseMoves } from '../stores/showcase.js'
  import { theme } from '../stores/theme.js'
  import { resolveThemeColors } from './chartHelpers.js'
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
  let markerColor = '#c8962e'
  function drawMarker(u) {
    if (markerPly == null) return
    const ctx = u.ctx
    const x = u.valToPos(markerPly, 'x', true)
    if (!isFinite(x)) return
    ctx.save()
    ctx.strokeStyle = markerColor
    ctx.lineWidth = 2
    ctx.setLineDash([3, 3])
    ctx.beginPath()
    ctx.moveTo(x, u.bbox.top)
    ctx.lineTo(x, u.bbox.top + u.bbox.height)
    ctx.stroke()
    ctx.restore()
  }

  function readPalette() {
    const c = resolveThemeColors()
    const root = getComputedStyle(document.documentElement)
    return {
      axis: c.axisColor || '#888',
      text: c.textColor || '#bbb',
      grid: c.gridColor || '#333',
      teal: root.getPropertyValue('--accent-teal').trim() || '#4db8a8',
      gold: root.getPropertyValue('--accent-gold').trim() || '#c8962e',
    }
  }

  function makeOpts(width, palette) {
    return {
      width,
      height: 120,
      cursor: { show: false },
      legend: { show: false },
      scales: { y: { range: [0, 1] } },
      axes: [
        { show: true, size: 20, font: '10px sans-serif', stroke: palette.axis,
          grid: { stroke: palette.grid, width: 0.5 },
          ticks: { stroke: palette.axis } },
        { show: true, size: 30, font: '10px sans-serif', stroke: palette.axis,
          grid: { stroke: palette.grid, width: 0.5 },
          ticks: { stroke: palette.axis },
          values: (u, vals) => vals.map(v => (v * 100).toFixed(0) + '%') },
      ],
      series: [
        {},
        { stroke: palette.teal, width: 2, fill: palette.teal + '1f' },
      ],
      hooks: { draw: [drawMarker] },
    }
  }

  function currentWidth() {
    if (!containerEl) return 300
    const w = containerEl.clientWidth
    return Math.max(120, w)
  }

  function rebuildChart() {
    if (!chartEl) return
    if (chart) {
      chart.destroy()
      chart = null
    }
    const palette = readPalette()
    markerColor = palette.gold
    chart = new uPlot(makeOpts(currentWidth(), palette), buildData($winProbHistory), chartEl)
  }

  // Click-to-scrub: translate canvas X position into ply index. uPlot exposes
  // posToVal() for converting pixel positions back into data values.
  function onChartClick(event) {
    if (!chart) return
    const rect = chart.over.getBoundingClientRect()
    const x = event.clientX - rect.left
    const ply = Math.round(chart.posToVal(x, 'x'))
    if (!Number.isFinite(ply)) return
    const moves = $showcaseMoves
    if (!moves || moves.length === 0) return
    const idx = moves.findIndex(m => m.ply === ply)
    const target = idx >= 0 ? idx : Math.max(0, Math.min(ply, moves.length - 1))
    if (target >= moves.length - 1) showcaseSelectedPly.set(null)
    else showcaseSelectedPly.set(target)
  }

  onMount(() => {
    rebuildChart()
    if (typeof ResizeObserver !== 'undefined' && containerEl) {
      resizeObserver = new ResizeObserver(() => {
        if (chart) chart.setSize({ width: currentWidth(), height: 120 })
      })
      resizeObserver.observe(containerEl)
    }
  })

  // Rebuild on theme change so axis/series strokes pick up the new palette.
  const unsubTheme = theme.subscribe(() => {
    if (chartEl) requestAnimationFrame(rebuildChart)
  })

  onDestroy(() => {
    unsubTheme()
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
  <div
    class="chart-host"
    bind:this={containerEl}
    on:click={onChartClick}
    on:keydown={(e) => { if (e.key === 'Enter') onChartClick(e) }}
    role="presentation"
  >
    <div bind:this={chartEl}></div>
  </div>
  <p class="axis-hint">Higher = side-to-move advantage. Click chart to jump.</p>
</div>

<style>
  .win-prob-graph { padding: 8px; }
  .section-label { font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin: 0 0 4px; }
  .chart-host { width: 100%; min-width: 0; cursor: pointer; }
  .axis-hint {
    margin: 4px 0 0;
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.3;
  }
</style>
