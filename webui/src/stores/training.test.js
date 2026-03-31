import { describe, it, expect, beforeEach, vi } from 'vitest'
import { get } from 'svelte/store'
import { trainingState, trainingAlive } from './training.js'

beforeEach(() => {
  trainingState.set(null)
  vi.useRealTimers()
})

describe('trainingState store', () => {
  it('starts null', () => {
    expect(get(trainingState)).toBeNull()
  })

  it('stores training state object', () => {
    const state = { model_arch: 'resnet', display_name: 'Bot', heartbeat_at: '' }
    trainingState.set(state)
    expect(get(trainingState)).toEqual(state)
  })
})

describe('trainingAlive derived store', () => {
  it('returns false when state is null', () => {
    expect(get(trainingAlive)).toBe(false)
  })

  it('returns false when heartbeat_at is missing', () => {
    trainingState.set({ model_arch: 'resnet' })
    expect(get(trainingAlive)).toBe(false)
  })

  it('returns false when heartbeat_at is empty string', () => {
    trainingState.set({ heartbeat_at: '' })
    expect(get(trainingAlive)).toBe(false)
  })

  it('returns true when heartbeat is fresh (< 30s)', () => {
    const now = new Date().toISOString()
    trainingState.set({ heartbeat_at: now })
    expect(get(trainingAlive)).toBe(true)
  })

  it('returns false when heartbeat is stale (> 30s)', () => {
    const stale = new Date(Date.now() - 60000).toISOString()
    trainingState.set({ heartbeat_at: stale })
    expect(get(trainingAlive)).toBe(false)
  })

  it('returns false when heartbeat is exactly at the boundary', () => {
    // 30001ms ago — just past the 30s threshold
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-04-01T12:00:30.001Z'))
    trainingState.set({ heartbeat_at: '2026-04-01T12:00:00.000Z' })
    expect(get(trainingAlive)).toBe(false)
  })

  it('returns true when heartbeat is just under the boundary', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-04-01T12:00:29.999Z'))
    trainingState.set({ heartbeat_at: '2026-04-01T12:00:00.000Z' })
    expect(get(trainingAlive)).toBe(true)
  })
})
