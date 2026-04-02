// @vitest-environment jsdom
import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { handleMessage } from './ws.js'
import { games, selectedGameId } from '../stores/games.js'
import { metrics } from '../stores/metrics.js'
import { trainingState } from '../stores/training.js'

beforeEach(() => {
  games.set([])
  selectedGameId.set(0) // matches store default (writable(0) in games.js)
  metrics.set([])
  trainingState.set(null)
})

describe('handleMessage — init', () => {
  it('populates all stores from init message', () => {
    handleMessage({
      type: 'init',
      games: [{ game_id: 0, board: '...' }],
      metrics: [{ id: 1, step: 100 }],
      training_state: { display_name: 'Bot', model_arch: 'resnet' },
    })

    expect(get(games)).toHaveLength(1)
    expect(get(metrics)).toHaveLength(1)
    expect(get(trainingState).display_name).toBe('Bot')
  })

  it('preserves selectedGameId when games are present and ID is already set', () => {
    selectedGameId.set(0)
    handleMessage({
      type: 'init',
      games: [{ game_id: 0 }],
      metrics: [],
      training_state: null,
    })
    expect(get(selectedGameId)).toBe(0)
  })

  it('sets selectedGameId to 0 when games are present and ID is null', () => {
    selectedGameId.set(null)
    handleMessage({
      type: 'init',
      games: [{ game_id: 0 }],
      metrics: [],
      training_state: null,
    })
    expect(get(selectedGameId)).toBe(0)
  })

  it('handles missing fields gracefully', () => {
    handleMessage({ type: 'init' })
    expect(get(games)).toEqual([])
    expect(get(metrics)).toEqual([])
    expect(get(trainingState)).toBeNull()
  })
})

describe('handleMessage — game_update', () => {
  it('replaces games list with snapshots', () => {
    games.set([{ game_id: 0, ply: 10 }])
    handleMessage({
      type: 'game_update',
      snapshots: [{ game_id: 0, ply: 20 }, { game_id: 1, ply: 5 }],
    })
    expect(get(games)).toHaveLength(2)
    expect(get(games)[0].ply).toBe(20)
  })

  it('handles missing snapshots field', () => {
    handleMessage({ type: 'game_update' })
    expect(get(games)).toEqual([])
  })
})

describe('handleMessage — metrics_update', () => {
  it('appends new rows to existing metrics', () => {
    metrics.set([{ id: 1, step: 100 }])
    handleMessage({
      type: 'metrics_update',
      rows: [{ id: 2, step: 200 }, { id: 3, step: 300 }],
    })
    expect(get(metrics)).toHaveLength(3)
    expect(get(metrics)[2].id).toBe(3)
  })

  it('handles missing rows field', () => {
    metrics.set([{ id: 1 }])
    handleMessage({ type: 'metrics_update' })
    expect(get(metrics)).toHaveLength(1)
  })
})

describe('handleMessage — training_status', () => {
  it('merges status update into existing state', () => {
    trainingState.set({ display_name: 'Bot', model_arch: 'resnet', status: 'running' })
    handleMessage({
      type: 'training_status',
      status: 'completed',
      heartbeat_at: '2026-04-01T12:00:00Z',
      epoch: 42,
    })
    const state = get(trainingState)
    expect(state.status).toBe('completed')
    expect(state.current_epoch).toBe(42)
    expect(state.display_name).toBe('Bot') // preserved
  })
})

describe('handleMessage — ping', () => {
  it('does not modify any stores', () => {
    games.set([{ game_id: 0 }])
    metrics.set([{ id: 1 }])
    trainingState.set({ status: 'running' })

    handleMessage({ type: 'ping' })

    expect(get(games)).toHaveLength(1)
    expect(get(metrics)).toHaveLength(1)
    expect(get(trainingState).status).toBe('running')
  })
})

describe('handleMessage — unknown type', () => {
  it('silently ignores unknown message types', () => {
    expect(() => handleMessage({ type: 'unknown_future_type' })).not.toThrow()
  })
})
