import { describe, it, expect } from 'vitest'
import { extractColumns } from './metricsColumns.js'

describe('extractColumns', () => {
  it('returns all empty arrays for empty input', () => {
    const cols = extractColumns([])
    expect(cols.steps).toEqual([])
    expect(cols.policyLoss).toEqual([])
    expect(cols.valueLoss).toEqual([])
    expect(cols.winRate).toEqual([])
    expect(cols.blackWinRate).toEqual([])
    expect(cols.whiteWinRate).toEqual([])
    expect(cols.drawRate).toEqual([])
    expect(cols.avgEpLen).toEqual([])
    expect(cols.entropy).toEqual([])
    expect(cols.epochs).toEqual([])
  })

  it('extracts a single fully-populated row', () => {
    const row = {
      step: 100,
      policy_loss: 0.5,
      value_loss: 0.3,
      win_rate: 0.6,
      black_win_rate: 0.7,
      white_win_rate: 0.4,
      draw_rate: 0.1,
      avg_episode_length: 42,
      entropy: 1.2,
      epoch: 3,
    }
    const cols = extractColumns([row])
    expect(cols.steps).toEqual([100])
    expect(cols.policyLoss).toEqual([0.5])
    expect(cols.valueLoss).toEqual([0.3])
    expect(cols.winRate).toEqual([0.6])
    expect(cols.blackWinRate).toEqual([0.7])
    expect(cols.whiteWinRate).toEqual([0.4])
    expect(cols.drawRate).toEqual([0.1])
    expect(cols.avgEpLen).toEqual([42])
    expect(cols.entropy).toEqual([1.2])
    expect(cols.epochs).toEqual([3])
  })

  it('handles multiple rows with partial fields (null coalescing)', () => {
    const rows = [
      { step: 1, policy_loss: 0.9, epoch: 1 },
      { step: 2, value_loss: 0.4, epoch: 2 },
    ]
    const cols = extractColumns(rows)
    expect(cols.steps).toEqual([1, 2])
    expect(cols.policyLoss).toEqual([0.9, null])
    expect(cols.valueLoss).toEqual([null, 0.4])
    expect(cols.winRate).toEqual([null, null])
    expect(cols.blackWinRate).toEqual([null, null])
    expect(cols.whiteWinRate).toEqual([null, null])
    expect(cols.drawRate).toEqual([null, null])
    expect(cols.avgEpLen).toEqual([null, null])
    expect(cols.entropy).toEqual([null, null])
    expect(cols.epochs).toEqual([1, 2])
  })

  it('defaults missing step to 0 (uses ||)', () => {
    const cols = extractColumns([{}])
    expect(cols.steps).toEqual([0])
  })

  it('defaults falsy step (0) to 0', () => {
    const cols = extractColumns([{ step: 0 }])
    expect(cols.steps).toEqual([0])
  })

  it('defaults missing epoch to 0 (uses ||)', () => {
    const cols = extractColumns([{}])
    expect(cols.epochs).toEqual([0])
  })

  it('defaults falsy epoch (0) to 0', () => {
    const cols = extractColumns([{ epoch: 0 }])
    expect(cols.epochs).toEqual([0])
  })

  it('defaults missing optional fields to null (uses ??)', () => {
    const cols = extractColumns([{ step: 5, epoch: 1 }])
    expect(cols.policyLoss).toEqual([null])
    expect(cols.valueLoss).toEqual([null])
    expect(cols.winRate).toEqual([null])
    expect(cols.blackWinRate).toEqual([null])
    expect(cols.whiteWinRate).toEqual([null])
    expect(cols.drawRate).toEqual([null])
    expect(cols.avgEpLen).toEqual([null])
    expect(cols.entropy).toEqual([null])
  })

  it('preserves explicit 0 for optional fields via ?? (not ||)', () => {
    const row = {
      step: 1,
      policy_loss: 0,
      value_loss: 0,
      win_rate: 0,
      black_win_rate: 0,
      white_win_rate: 0,
      draw_rate: 0,
      avg_episode_length: 0,
      entropy: 0,
      epoch: 1,
    }
    const cols = extractColumns([row])
    expect(cols.policyLoss).toEqual([0])
    expect(cols.valueLoss).toEqual([0])
    expect(cols.winRate).toEqual([0])
    expect(cols.blackWinRate).toEqual([0])
    expect(cols.whiteWinRate).toEqual([0])
    expect(cols.drawRate).toEqual([0])
    expect(cols.avgEpLen).toEqual([0])
    expect(cols.entropy).toEqual([0])
  })

  it('reads snake_case field names from rows', () => {
    const row = {
      step: 10,
      policy_loss: 1.1,
      value_loss: 2.2,
      win_rate: 0.5,
      black_win_rate: 0.6,
      white_win_rate: 0.3,
      draw_rate: 0.1,
      avg_episode_length: 50,
      entropy: 0.8,
      epoch: 2,
    }
    // Verify camelCase output keys map from snake_case input
    const cols = extractColumns([row])
    expect(cols.policyLoss).toEqual([1.1])
    expect(cols.valueLoss).toEqual([2.2])
    expect(cols.blackWinRate).toEqual([0.6])
    expect(cols.whiteWinRate).toEqual([0.3])
    expect(cols.avgEpLen).toEqual([50])
  })
})
