/**
 * Read resolved CSS custom properties from the document root.
 * Canvas 2D context cannot use var() references, so we must
 * resolve them to concrete color strings before passing to uPlot.
 */
export function resolveThemeColors() {
  const style = getComputedStyle(document.documentElement)
  return {
    textColor: style.getPropertyValue('--text-secondary').trim(),
    gridColor: style.getPropertyValue('--border-subtle').trim(),
    axisColor: style.getPropertyValue('--text-muted').trim(),
  }
}

/**
 * Build uPlot options object.
 * @param {object} params
 * @param {number} params.width
 * @param {number} params.height
 * @param {Array<{label: string, color: string}>} params.series
 * @param {boolean} [params.compact]
 * @param {{textColor: string, gridColor: string, axisColor: string}} [params.colors]
 * @returns {object} uPlot options
 */
export function buildChartOpts({ width, height, series, compact = false, colors = null }) {
  const c = colors || resolveThemeColors()
  return {
    width,
    height,
    padding: compact ? [4, 4, 0, 0] : [8, 8, 0, 0],
    cursor: { show: !compact },
    legend: { show: !compact },
    scales: { x: { time: false } },
    axes: [
      {
        show: !compact,
        stroke: c.textColor,
        grid: { stroke: c.gridColor, width: 0.5 },
        ticks: { stroke: c.axisColor },
        font: '12px sans-serif',
        values: (u, vals) => vals.map(v => Number.isInteger(v) ? v : ''),
        incrs: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
      },
      {
        show: !compact,
        stroke: c.textColor,
        grid: { stroke: c.gridColor, width: 0.5 },
        ticks: { stroke: c.axisColor },
        font: '12px sans-serif',
      },
    ],
    series: [
      { label: 'X' },
      ...series.map(s => ({
        label: s.label,
        stroke: s.color,
        width: compact ? 1 : 1.5,
        fill: s.color + '20',
      })),
    ],
  }
}

/**
 * Build uPlot data array from x-axis data and series.
 * @param {number[]} xData
 * @param {Array<{data: number[]}>} series
 * @returns {Array<number[]>}
 */
export function buildChartData(xData, series) {
  return [xData, ...series.map(s => s.data)]
}
