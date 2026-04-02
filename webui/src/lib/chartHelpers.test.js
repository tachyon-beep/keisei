import { describe, it, expect } from 'vitest'
import { DARK_THEME, buildChartOpts, buildChartData } from './chartHelpers.js'

describe('buildChartData', () => {
  it('returns array with empty xData and no series', () => {
    const result = buildChartData([], [])
    expect(result).toEqual([[]])
  })

  it('returns xData with a single series', () => {
    const xData = [1, 2, 3]
    const series = [{ data: [10, 20, 30] }]
    const result = buildChartData(xData, series)
    expect(result).toEqual([[1, 2, 3], [10, 20, 30]])
  })

  it('returns xData with multiple series', () => {
    const xData = [1, 2]
    const series = [
      { data: [10, 20] },
      { data: [30, 40] },
      { data: [50, 60] },
    ]
    const result = buildChartData(xData, series)
    expect(result).toEqual([[1, 2], [10, 20], [30, 40], [50, 60]])
  })

  it('passes data arrays by reference, not by copy', () => {
    const xData = [1, 2, 3]
    const seriesData = [10, 20, 30]
    const series = [{ data: seriesData }]
    const result = buildChartData(xData, series)
    expect(result[0]).toBe(xData)
    expect(result[1]).toBe(seriesData)
  })
})

describe('buildChartOpts', () => {
  const defaultParams = {
    width: 800,
    height: 200,
    series: [
      { label: 'Loss', color: '#ff0000' },
      { label: 'Accuracy', color: '#00ff00' },
    ],
  }

  it('returns correct width and height', () => {
    const opts = buildChartOpts(defaultParams)
    expect(opts.width).toBe(800)
    expect(opts.height).toBe(200)
  })

  it('maps series with correct labels, stroke colors, width, and fill', () => {
    const opts = buildChartOpts(defaultParams)
    // Skip first entry (the X label)
    const mapped = opts.series.slice(1)
    expect(mapped).toEqual([
      { label: 'Loss', stroke: '#ff0000', width: 1.5, fill: '#ff000020' },
      { label: 'Accuracy', stroke: '#00ff00', width: 1.5, fill: '#00ff0020' },
    ])
  })

  it('always includes X label as first series entry', () => {
    const opts = buildChartOpts(defaultParams)
    expect(opts.series[0]).toEqual({ label: 'X' })
  })

  it('includes X label even with empty series', () => {
    const opts = buildChartOpts({ width: 400, height: 100, series: [] })
    expect(opts.series).toEqual([{ label: 'X' }])
  })

  it('sets padding to [8, 8, 0, 0]', () => {
    const opts = buildChartOpts(defaultParams)
    expect(opts.padding).toEqual([8, 8, 0, 0])
  })

  it('sets x scale with time: false', () => {
    const opts = buildChartOpts(defaultParams)
    expect(opts.scales.x.time).toBe(false)
  })

  it('uses DARK_THEME colors for axes', () => {
    const opts = buildChartOpts(defaultParams)
    const [xAxis, yAxis] = opts.axes

    expect(xAxis.stroke).toBe(DARK_THEME.textColor)
    expect(xAxis.grid.stroke).toBe(DARK_THEME.gridColor)
    expect(xAxis.ticks.stroke).toBe(DARK_THEME.axisColor)

    expect(yAxis.stroke).toBe(DARK_THEME.textColor)
    expect(yAxis.grid.stroke).toBe(DARK_THEME.gridColor)
    expect(yAxis.ticks.stroke).toBe(DARK_THEME.axisColor)
  })

  it('x-axis values filter passes integers and blanks non-integers', () => {
    const opts = buildChartOpts(defaultParams)
    const valuesFn = opts.axes[0].values
    const result = valuesFn(null, [1, 2.5, 3, 4.1, 0, -1, 1.0])
    expect(result).toEqual([1, '', 3, '', 0, -1, 1.0])
  })

  it('x-axis incrs array is correct', () => {
    const opts = buildChartOpts(defaultParams)
    expect(opts.axes[0].incrs).toEqual([
      1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
    ])
  })
})
