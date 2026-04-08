import { describe, it, expect } from 'vitest'
import { buildChartOpts, buildChartData } from './chartHelpers.js'

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
  const testColors = {
    textColor: '#a89880',
    gridColor: '#221e18',
    axisColor: '#9a8878',
  }

  const defaultParams = {
    width: 800,
    height: 200,
    series: [
      { label: 'Loss', color: '#ff0000' },
      { label: 'Accuracy', color: '#00ff00' },
    ],
    colors: testColors,
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
      { label: 'Loss', stroke: '#ff0000', width: 1.5, fill: '#ff000020', dash: undefined, scale: 'y' },
      { label: 'Accuracy', stroke: '#00ff00', width: 1.5, fill: '#00ff0020', dash: undefined, scale: 'y' },
    ])
  })

  it('always includes X label as first series entry', () => {
    const opts = buildChartOpts(defaultParams)
    expect(opts.series[0]).toEqual({ label: 'X' })
  })

  it('includes X label even with empty series', () => {
    const opts = buildChartOpts({ width: 400, height: 100, series: [], colors: testColors })
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

  it('uses provided colors for axes', () => {
    const opts = buildChartOpts(defaultParams)
    const [xAxis, yAxis] = opts.axes

    expect(xAxis.stroke).toBe(testColors.textColor)
    expect(xAxis.grid.stroke).toBe(testColors.gridColor)
    expect(xAxis.ticks.stroke).toBe(testColors.axisColor)

    expect(yAxis.stroke).toBe(testColors.textColor)
    expect(yAxis.grid.stroke).toBe(testColors.gridColor)
    expect(yAxis.ticks.stroke).toBe(testColors.axisColor)
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

