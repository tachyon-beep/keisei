// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { get } from 'svelte/store'

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
})

describe('audio store', () => {
  it('defaults to false when nothing stored', async () => {
    const { audioEnabled } = await import('./audio.js')
    expect(get(audioEnabled)).toBe(false)
  })

  it('rehydrates a stored "true"', async () => {
    localStorage.setItem('audioEnabled', 'true')
    const { audioEnabled } = await import('./audio.js')
    expect(get(audioEnabled)).toBe(true)
  })

  it('rehydrates a stored "false"', async () => {
    localStorage.setItem('audioEnabled', 'false')
    const { audioEnabled } = await import('./audio.js')
    expect(get(audioEnabled)).toBe(false)
  })

  it('falls back to false for malformed values', async () => {
    localStorage.setItem('audioEnabled', 'banana')
    const { audioEnabled } = await import('./audio.js')
    expect(get(audioEnabled)).toBe(false)
  })

  it('persists true to localStorage on set', async () => {
    const { audioEnabled } = await import('./audio.js')
    audioEnabled.set(true)
    expect(localStorage.getItem('audioEnabled')).toBe('true')
  })

  it('persists false to localStorage on set', async () => {
    localStorage.setItem('audioEnabled', 'true')
    const { audioEnabled } = await import('./audio.js')
    audioEnabled.set(false)
    expect(localStorage.getItem('audioEnabled')).toBe('false')
  })

  it('exposes AUDIO_VOLUME as a number in [0, 1]', async () => {
    const { AUDIO_VOLUME } = await import('./audio.js')
    expect(typeof AUDIO_VOLUME).toBe('number')
    expect(AUDIO_VOLUME).toBeGreaterThanOrEqual(0)
    expect(AUDIO_VOLUME).toBeLessThanOrEqual(1)
  })
})
