// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { get } from 'svelte/store'

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
})

describe('aboutLevel store', () => {
  it('defaults to level 2 (Learning Loop) when nothing stored', async () => {
    const { aboutLevel } = await import('./aboutLevel.js')
    expect(get(aboutLevel)).toBe(2)
  })

  it('rehydrates a valid stored level', async () => {
    localStorage.setItem('aboutLevel', '4')
    const { aboutLevel } = await import('./aboutLevel.js')
    expect(get(aboutLevel)).toBe(4)
  })

  it('rehydrates the new top level (5)', async () => {
    localStorage.setItem('aboutLevel', '5')
    const { aboutLevel } = await import('./aboutLevel.js')
    expect(get(aboutLevel)).toBe(5)
  })

  it('falls back to default for out-of-range values', async () => {
    localStorage.setItem('aboutLevel', '99')
    const { aboutLevel } = await import('./aboutLevel.js')
    expect(get(aboutLevel)).toBe(2)
  })

  it('falls back to default for the now-out-of-range value 6', async () => {
    localStorage.setItem('aboutLevel', '6')
    const { aboutLevel } = await import('./aboutLevel.js')
    expect(get(aboutLevel)).toBe(2)
  })

  it('falls back to default for non-numeric values', async () => {
    localStorage.setItem('aboutLevel', 'banana')
    const { aboutLevel } = await import('./aboutLevel.js')
    expect(get(aboutLevel)).toBe(2)
  })

  it('persists set values to localStorage', async () => {
    const { aboutLevel } = await import('./aboutLevel.js')
    aboutLevel.set(3)
    expect(localStorage.getItem('aboutLevel')).toBe('3')
  })

  it('exposes ABOUT_LEVELS metadata in order', async () => {
    const { ABOUT_LEVELS } = await import('./aboutLevel.js')
    expect(ABOUT_LEVELS.map((l) => l.id)).toEqual([1, 2, 3, 4, 5])
    expect(ABOUT_LEVELS.map((l) => l.label)).toEqual([
      'The Big Idea',
      'Learning Loop',
      'Inside the Demo',
      'Algorithmic',
      'Research View',
    ])
  })
})
