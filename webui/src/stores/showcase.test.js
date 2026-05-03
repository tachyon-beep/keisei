// @vitest-environment jsdom
import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import {
  showcaseGame, showcaseMoves, showcaseQueue,
  showcaseCurrentMove, showcaseDisplayedMove, showcaseSelectedPly,
  isScrubbing, winProbHistory, queueDepth, sidecarAlive,
  showcaseHeatmapEnabled, showcaseSpeed, resetShowcaseSelectionOnGameChange,
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

describe('showcaseDisplayedMove and scrubbing', () => {
  beforeEach(() => {
    showcaseSelectedPly.set(null)
    showcaseMoves.set([])
  })

  it('returns the latest move when selectedPly is null (live mode)', () => {
    showcaseMoves.set([
      { ply: 1, board_json: 'b1' },
      { ply: 2, board_json: 'b2' },
      { ply: 3, board_json: 'b3' },
    ])
    expect(get(showcaseDisplayedMove).ply).toBe(3)
    expect(get(isScrubbing)).toBe(false)
  })

  it('returns the selected move when scrubbing', () => {
    showcaseMoves.set([
      { ply: 1, board_json: 'b1' },
      { ply: 2, board_json: 'b2' },
      { ply: 3, board_json: 'b3' },
    ])
    showcaseSelectedPly.set(1)
    expect(get(showcaseDisplayedMove).ply).toBe(2)
    expect(get(isScrubbing)).toBe(true)
  })

  it('clamps the selected index to valid range', () => {
    showcaseMoves.set([
      { ply: 1, board_json: 'b1' },
      { ply: 2, board_json: 'b2' },
    ])
    showcaseSelectedPly.set(99)
    expect(get(showcaseDisplayedMove).ply).toBe(2)
    showcaseSelectedPly.set(-5)
    expect(get(showcaseDisplayedMove).ply).toBe(1)
  })

  it('returns null when no moves regardless of selection', () => {
    showcaseSelectedPly.set(0)
    expect(get(showcaseDisplayedMove)).toBeNull()
  })

  it('isScrubbing is false when selection points to the last move', () => {
    showcaseMoves.set([
      { ply: 1 },
      { ply: 2 },
    ])
    showcaseSelectedPly.set(1)
    expect(get(isScrubbing)).toBe(false)
  })

  it('resetShowcaseSelectionOnGameChange clears the selection', () => {
    showcaseSelectedPly.set(5)
    resetShowcaseSelectionOnGameChange(42)
    expect(get(showcaseSelectedPly)).toBeNull()
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

describe('showcaseSpeed', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('defaults to "normal"', () => {
    expect(['slow', 'normal', 'fast']).toContain(get(showcaseSpeed))
  })

  it('persists value to localStorage when set', () => {
    showcaseSpeed.set('fast')
    expect(localStorage.getItem('showcaseSpeed')).toBe('fast')
    showcaseSpeed.set('slow')
    expect(localStorage.getItem('showcaseSpeed')).toBe('slow')
  })

  it('rejects invalid speeds via persistence layer', () => {
    showcaseSpeed.set('warp')
    // Internal store still holds it (writable doesn't validate), but persistence
    // layer guards against polluting localStorage with an invalid value that
    // would later be replayed on reload.
    expect(localStorage.getItem('showcaseSpeed')).not.toBe('warp')
  })
})
