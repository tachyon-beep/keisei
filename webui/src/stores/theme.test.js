// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { get } from 'svelte/store'

// Each test needs a fresh module to test localStorage-at-load behavior.
beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
  document.documentElement.removeAttribute('data-theme')
})

describe('theme store', () => {
  it('defaults to "dark" when localStorage is empty', async () => {
    const { theme } = await import('./theme.js')
    expect(get(theme)).toBe('dark')
  })

  it('rehydrates from localStorage when a value is stored', async () => {
    localStorage.setItem('keisei-theme', 'light')
    const { theme } = await import('./theme.js')
    expect(get(theme)).toBe('light')
  })

  it('writes to localStorage on change', async () => {
    const { theme } = await import('./theme.js')
    theme.set('light')
    expect(localStorage.getItem('keisei-theme')).toBe('light')
  })

  it('sets data-theme attribute on document.documentElement', async () => {
    const { theme } = await import('./theme.js')
    // Initial subscription fires with default 'dark'
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
    theme.set('light')
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
  })
})

describe('toggleTheme', () => {
  it('toggles dark to light', async () => {
    const { theme, toggleTheme } = await import('./theme.js')
    expect(get(theme)).toBe('dark')
    toggleTheme()
    expect(get(theme)).toBe('light')
  })

  it('toggles light to dark', async () => {
    localStorage.setItem('keisei-theme', 'light')
    const { theme, toggleTheme } = await import('./theme.js')
    expect(get(theme)).toBe('light')
    toggleTheme()
    expect(get(theme)).toBe('dark')
  })

  it('full cycle: dark → light → dark', async () => {
    const { theme, toggleTheme } = await import('./theme.js')
    expect(get(theme)).toBe('dark')
    toggleTheme()
    expect(get(theme)).toBe('light')
    toggleTheme()
    expect(get(theme)).toBe('dark')
  })

  it('persists toggle result to localStorage and DOM', async () => {
    const { toggleTheme } = await import('./theme.js')
    toggleTheme()
    expect(localStorage.getItem('keisei-theme')).toBe('light')
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
  })
})
