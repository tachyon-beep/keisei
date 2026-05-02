// @vitest-environment jsdom
import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import {
  showcaseGame, showcaseMoves, showcaseQueue,
  showcaseCurrentMove, winProbHistory, queueDepth, sidecarAlive,
  showcaseHeatmapEnabled,
} from './showcase.js'

describe('showcase stores', () => {
  it('showcaseCurrentMove returns null when no moves (not board — full move obj)', () => {
    showcaseMoves.set([])
    expect(get(showcaseCurrentMove)).toBeNull()
  })

  it('showcaseCurrentMove returns latest move', () => {
    showcaseMoves.set([
      { ply: 1, board_json: 'b1', value_estimate: 0.5 },
      { ply: 2, board_json: 'b2', value_estimate: 0.6 },
    ])
    expect(get(showcaseCurrentMove).ply).toBe(2)
  })

  it('winProbHistory maps moves to ply/value pairs', () => {
    showcaseMoves.set([
      { ply: 1, value_estimate: 0.5 },
      { ply: 2, value_estimate: 0.7 },
    ])
    const history = get(winProbHistory)
    expect(history).toEqual([
      { ply: 1, value: 0.5 },
      { ply: 2, value: 0.7 },
    ])
  })

  it('queueDepth counts pending entries', () => {
    showcaseQueue.set([
      { id: 1, status: 'running' },
      { id: 2, status: 'pending' },
      { id: 3, status: 'pending' },
    ])
    expect(get(queueDepth)).toBe(2)
  })
})

describe('showcaseHeatmapEnabled', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('exposes a boolean store', () => {
    expect(typeof get(showcaseHeatmapEnabled)).toBe('boolean')
  })

  it('persists value to localStorage when toggled', () => {
    showcaseHeatmapEnabled.set(true)
    expect(localStorage.getItem('showcaseHeatmapEnabled')).toBe('true')
    showcaseHeatmapEnabled.set(false)
    expect(localStorage.getItem('showcaseHeatmapEnabled')).toBe('false')
  })
})
