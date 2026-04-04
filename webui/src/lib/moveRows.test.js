import { describe, it, expect } from 'vitest'
import { parseMoves, buildMoveRows, toJapanese } from './moveRows.js'

describe('parseMoves', () => {
  it('parses valid JSON string', () => {
    const result = parseMoves('[{"notation":"P-76"}]')
    expect(result).toEqual([{ notation: 'P-76' }])
  })

  it('returns empty array for invalid JSON', () => {
    expect(parseMoves('{broken')).toEqual([])
  })

  it('returns empty array for empty string', () => {
    expect(parseMoves('[]')).toEqual([])
  })

  it('passes through arrays as-is', () => {
    const arr = [{ notation: 'P-76' }]
    expect(parseMoves(arr)).toBe(arr)
  })

  it('returns empty array for null', () => {
    expect(parseMoves(null)).toEqual([])
  })

  it('returns empty array for undefined', () => {
    expect(parseMoves(undefined)).toEqual([])
  })
})

describe('buildMoveRows', () => {
  it('returns empty array for no moves', () => {
    expect(buildMoveRows([])).toEqual([])
  })

  it('pairs moves into rows', () => {
    const moves = [
      { notation: 'P-76' },
      { notation: 'P-34' },
      { notation: 'P-26' },
      { notation: 'P-84' },
    ]
    const rows = buildMoveRows(moves)
    expect(rows).toEqual([
      { num: 1, black: 'P-76', white: 'P-34', isLatest: false },
      { num: 2, black: 'P-26', white: 'P-84', isLatest: true },
    ])
  })

  it('handles odd number of moves (black moved last)', () => {
    const moves = [
      { notation: 'P-76' },
      { notation: 'P-34' },
      { notation: 'P-26' },
    ]
    const rows = buildMoveRows(moves)
    expect(rows).toEqual([
      { num: 1, black: 'P-76', white: 'P-34', isLatest: false },
      { num: 2, black: 'P-26', white: '', isLatest: true },
    ])
  })

  it('single move marks first row as latest', () => {
    const rows = buildMoveRows([{ notation: 'P-76' }])
    expect(rows).toEqual([
      { num: 1, black: 'P-76', white: '', isLatest: true },
    ])
  })

  it('handles moves without notation field', () => {
    const rows = buildMoveRows([{}, { notation: 'P-34' }])
    expect(rows).toEqual([
      { num: 1, black: '', white: 'P-34', isLatest: true },
    ])
  })
})

describe('toJapanese', () => {
  it('converts simple Hodges move', () => {
    expect(toJapanese('P-7f')).toBe('P-７六')
  })

  it('converts capture notation', () => {
    expect(toJapanese('Bx3c')).toBe('Bx３三')
  })

  it('converts promoted piece move', () => {
    expect(toJapanese('+R-5a')).toBe('+R-５一')
  })

  it('converts drop notation', () => {
    expect(toJapanese('P*5e')).toBe('P*５五')
  })

  it('converts promotion suffix', () => {
    expect(toJapanese('Nx7c+')).toBe('Nx７三+')
  })

  it('converts declined promotion suffix', () => {
    expect(toJapanese('S-4d=')).toBe('S-４四=')
  })

  it('converts disambiguated move', () => {
    expect(toJapanese('G6g-5h')).toBe('G６七-５八')
  })

  it('returns empty string for empty input', () => {
    expect(toJapanese('')).toBe('')
  })
})
