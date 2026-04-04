import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { metrics, latestMetrics, appendMetrics } from './metrics.js'

beforeEach(() => {
  metrics.set([])
})

describe('metrics store', () => {
  it('starts empty', () => {
    expect(get(metrics)).toEqual([])
  })

  it('stores metric rows', () => {
    const rows = [
      { id: 1, epoch: 0, step: 100, policy_loss: 1.5 },
      { id: 2, epoch: 0, step: 200, policy_loss: 1.2 },
    ]
    metrics.set(rows)
    expect(get(metrics)).toEqual(rows)
  })

  it('prunes to MAX_POINTS (10000) when exceeded', () => {
    const big = Array.from({ length: 10050 }, (_, i) => ({ id: i, step: i }))
    metrics.set(big)
    const result = get(metrics)
    expect(result).toHaveLength(10000)
    // Should keep the LAST 10000 (newest data)
    expect(result[0].id).toBe(50)
    expect(result[result.length - 1].id).toBe(10049)
  })

  it('does not prune at exactly MAX_POINTS', () => {
    const exact = Array.from({ length: 10000 }, (_, i) => ({ id: i }))
    metrics.set(exact)
    expect(get(metrics)).toHaveLength(10000)
  })

  it('supports incremental update pattern', () => {
    metrics.set([{ id: 1 }])
    metrics.update(current => [...current, { id: 2 }, { id: 3 }])
    expect(get(metrics)).toHaveLength(3)
  })
})

describe('appendMetrics', () => {
  it('appends rows to existing metrics', () => {
    metrics.set([{ id: 1, step: 100 }])
    appendMetrics([{ id: 2, step: 200 }])
    const result = get(metrics)
    expect(result).toHaveLength(2)
    expect(result[1]).toEqual({ id: 2, step: 200 })
  })

  it('handles empty array (no-op)', () => {
    metrics.set([{ id: 1 }])
    appendMetrics([])
    expect(get(metrics)).toHaveLength(1)
  })

  it('triggers pruning when total exceeds MAX_POINTS', () => {
    // Fill to 9995 rows
    const existing = Array.from({ length: 9995 }, (_, i) => ({ id: i, step: i }))
    metrics.set(existing)
    // Append 10 more → total 10005 > 10000 → prune to last 10000
    const newRows = Array.from({ length: 10 }, (_, i) => ({ id: 10000 + i, step: 10000 + i }))
    appendMetrics(newRows)
    const result = get(metrics)
    expect(result).toHaveLength(10000)
    // Oldest rows should be dropped
    expect(result[0].id).toBe(5)
    expect(result[result.length - 1].id).toBe(10009)
  })
})

describe('latestMetrics derived store', () => {
  it('returns null when empty', () => {
    expect(get(latestMetrics)).toBeNull()
  })

  it('returns the last row', () => {
    metrics.set([
      { id: 1, step: 100 },
      { id: 2, step: 200 },
      { id: 3, step: 300 },
    ])
    expect(get(latestMetrics)).toEqual({ id: 3, step: 300 })
  })
})
