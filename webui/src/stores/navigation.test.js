// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { get } from 'svelte/store'

// Each test needs a fresh module to test localStorage-at-load behavior,
// so we use vi.resetModules() + dynamic import.

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
})

describe('activeTab store', () => {
  it('defaults to "training" when localStorage is empty', async () => {
    const { activeTab } = await import('./navigation.js')
    expect(get(activeTab)).toBe('training')
  })

  it('rehydrates from localStorage when a value is stored', async () => {
    localStorage.setItem('activeTab', 'league')
    const { activeTab } = await import('./navigation.js')
    expect(get(activeTab)).toBe('league')
  })

  it('writes back to localStorage on subscription', async () => {
    const { activeTab } = await import('./navigation.js')
    activeTab.set('games')
    expect(localStorage.getItem('activeTab')).toBe('games')
  })

  it('persists across set calls', async () => {
    const { activeTab } = await import('./navigation.js')
    activeTab.set('league')
    activeTab.set('training')
    expect(localStorage.getItem('activeTab')).toBe('training')
    expect(get(activeTab)).toBe('training')
  })
})
