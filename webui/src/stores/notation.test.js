// @vitest-environment jsdom
import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { notationStyle, NOTATION_STYLES, cycleNotation } from './notation.js'

describe('notationStyle store', () => {
  beforeEach(() => {
    localStorage.clear()
    notationStyle.set('western')
  })

  it('defaults to a valid style', () => {
    expect(NOTATION_STYLES).toContain(get(notationStyle))
  })

  it('persists value to localStorage when set', () => {
    notationStyle.set('japanese')
    expect(localStorage.getItem('notationStyle')).toBe('japanese')
    notationStyle.set('usi')
    expect(localStorage.getItem('notationStyle')).toBe('usi')
  })

  it('cycleNotation rotates through styles in order', () => {
    notationStyle.set('western')
    cycleNotation()
    expect(get(notationStyle)).toBe('japanese')
    cycleNotation()
    expect(get(notationStyle)).toBe('usi')
    cycleNotation()
    expect(get(notationStyle)).toBe('western')
  })

  it('persistence layer rejects invalid styles', () => {
    notationStyle.set('klingon')
    expect(localStorage.getItem('notationStyle')).not.toBe('klingon')
  })
})
