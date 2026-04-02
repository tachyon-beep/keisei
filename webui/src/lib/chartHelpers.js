/**
 * Dark theme for chart rendering.
 */
export const DARK_THEME = {
  background: 'var(--bg-primary)',
  gridColor: 'var(--border-subtle)',
  textColor: 'var(--text-secondary)',
  axisColor: 'var(--text-muted)',
}

/**
 * Build uPlot options object.
 * @param {object} params
 * @param {number} params.width
 * @param {number} params.height
 * @param {Array<{label: string, color: string}>} params.series
 * @returns {object} uPlot options
 */
export function buildChartOpts({ width, height, series }) {
  return {
    width,
    height,
    padding: [8, 8, 0, 0],
    cursor: { show: true },
    legend: { show: true },
    scales: { x: { time: false } },
    axes: [
      {
        stroke: DARK_THEME.textColor,
        grid: { stroke: DARK_THEME.gridColor, width: 0.5 },
        ticks: { stroke: DARK_THEME.axisColor },
        font: '12px sans-serif',
        values: (u, vals) => vals.map(v => Number.isInteger(v) ? v : ''),
        incrs: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
      },
      {
        stroke: DARK_THEME.textColor,
        grid: { stroke: DARK_THEME.gridColor, width: 0.5 },
        ticks: { stroke: DARK_THEME.axisColor },
        font: '12px sans-serif',
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

/**
 * Build uPlot data array from x-axis data and series.
 * @param {number[]} xData
 * @param {Array<{data: number[]}>} series
 * @returns {Array<number[]>}
 */
export function buildChartData(xData, series) {
  return [xData, ...series.map(s => s.data)]
}
