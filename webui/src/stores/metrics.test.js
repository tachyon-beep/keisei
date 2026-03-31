import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { metrics, latestMetrics } from './metrics.js'

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
