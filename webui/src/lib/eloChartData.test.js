import { describe, it, expect } from 'vitest'
import { buildEloChartData } from './eloChartData.js'

describe('buildEloChartData', () => {
  it('returns empty structure for empty input', () => {
    const result = buildEloChartData([], [])
    expect(result.xData).toEqual([])
    expect(result.series).toEqual([])
  })

  it('groups single entry into one series', () => {
    const history = [
      { entry_id: 1, epoch: 5, elo_rating: 1000 },
      { entry_id: 1, epoch: 10, elo_rating: 1050 },
    ]
    const entries = [{ id: 1, architecture: 'resnet', elo_rating: 1050 }]
    const result = buildEloChartData(history, entries)
    expect(result.xData).toEqual([5, 10])
    expect(result.series).toHaveLength(1)
    expect(result.series[0].label).toBe('resnet (1050)')
    expect(result.series[0].data).toEqual([1000, 1050])
  })

  it('groups multiple entries with shared epoch axis', () => {
    const history = [
      { entry_id: 1, epoch: 5, elo_rating: 1000 },
      { entry_id: 2, epoch: 5, elo_rating: 900 },
      { entry_id: 1, epoch: 10, elo_rating: 1050 },
      { entry_id: 2, epoch: 10, elo_rating: 950 },
    ]
    const entries = [
      { id: 1, architecture: 'resnet', elo_rating: 1050 },
      { id: 2, architecture: 'transformer', elo_rating: 950 },
    ]
    const result = buildEloChartData(history, entries)
    expect(result.xData).toEqual([5, 10])
    expect(result.series).toHaveLength(2)
    expect(result.series[0].data).toEqual([1000, 1050])
    expect(result.series[1].data).toEqual([900, 950])
  })

  it('fills null for epochs where an entry has no data', () => {
    const history = [
      { entry_id: 1, epoch: 5, elo_rating: 1000 },
      { entry_id: 1, epoch: 10, elo_rating: 1050 },
      { entry_id: 2, epoch: 10, elo_rating: 900 },
    ]
    const entries = [
      { id: 1, architecture: 'a', elo_rating: 1050 },
      { id: 2, architecture: 'b', elo_rating: 900 },
    ]
    const result = buildEloChartData(history, entries)
    expect(result.xData).toEqual([5, 10])
    expect(result.series[0].data).toEqual([1000, 1050])
    expect(result.series[1].data).toEqual([null, 900])
  })
})
