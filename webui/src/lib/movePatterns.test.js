import { describe, it, expect } from 'vitest'
import { PATTERNS, KING, ROOK, BISHOP, DRAGON, HORSE, GOLD, SILVER, KNIGHT, LANCE, PAWN } from './movePatterns.js'

describe('movePatterns', () => {
  it('all patterns are 3 rows of 3 columns', () => {
    const grids = [KING, ROOK, BISHOP, DRAGON, HORSE, GOLD, SILVER, KNIGHT, LANCE, PAWN]
    for (const g of grids) {
      expect(g).toHaveLength(3)
      for (const row of g) {
        expect(row).toHaveLength(3)
      }
    }
  })

  it('centre cell is always null (piece position)', () => {
    const grids = [KING, ROOK, BISHOP, DRAGON, HORSE, GOLD, SILVER, KNIGHT, LANCE, PAWN]
    for (const g of grids) {
      expect(g[1][1]).toBeNull()
    }
  })

  it('King can move to all 8 adjacent squares', () => {
    const steps = KING.flat().filter(c => c === 'step')
    expect(steps).toHaveLength(8)
  })

  it('Rook slides in 4 cardinal directions', () => {
    const slides = ROOK.flat().filter(c => c === 'slide')
    expect(slides).toHaveLength(4)
    expect(ROOK[0][1]).toBe('slide') // up
    expect(ROOK[2][1]).toBe('slide') // down
    expect(ROOK[1][0]).toBe('slide') // left
    expect(ROOK[1][2]).toBe('slide') // right
  })

  it('Bishop slides in 4 diagonal directions', () => {
    const slides = BISHOP.flat().filter(c => c === 'slide')
    expect(slides).toHaveLength(4)
  })

  it('Dragon = Rook slides + diagonal steps', () => {
    expect(DRAGON[0][1]).toBe('slide') // cardinal = slide
    expect(DRAGON[0][0]).toBe('step')  // diagonal = step
  })

  it('Horse = Bishop slides + orthogonal steps', () => {
    expect(HORSE[0][0]).toBe('slide')  // diagonal = slide
    expect(HORSE[0][1]).toBe('step')   // orthogonal = step
  })

  it('Gold moves to 6 squares (not back-diagonals)', () => {
    const steps = GOLD.flat().filter(c => c === 'step')
    expect(steps).toHaveLength(6)
    expect(GOLD[2][0]).toBeNull() // back-left diagonal
    expect(GOLD[2][2]).toBeNull() // back-right diagonal
  })

  it('Silver moves to 5 squares (forward 3 + back diagonals)', () => {
    const steps = SILVER.flat().filter(c => c === 'step')
    expect(steps).toHaveLength(5)
  })

  it('Knight has two forward jumps in row 0 corners and nothing else', () => {
    const jumps = KNIGHT.flat().filter(c => c === 'jump')
    expect(jumps).toHaveLength(2)
    expect(KNIGHT[0][0]).toBe('jump')
    expect(KNIGHT[0][2]).toBe('jump')
    expect(KNIGHT[0][1]).toBeNull()
    expect(KNIGHT[2].every(c => c === null)).toBe(true)
    // Knight no longer carries an extra-row escape hatch — legend stays 3 rows.
    expect(PATTERNS.Knight.extra).toBeUndefined()
  })

  it('Lance slides forward only', () => {
    const slides = LANCE.flat().filter(c => c === 'slide')
    expect(slides).toHaveLength(1)
    expect(LANCE[0][1]).toBe('slide')
  })

  it('Pawn steps forward only', () => {
    const steps = PAWN.flat().filter(c => c === 'step')
    expect(steps).toHaveLength(1)
    expect(PAWN[0][1]).toBe('step')
  })

  it('PATTERNS has entries for all 8 pieces', () => {
    expect(Object.keys(PATTERNS)).toHaveLength(8)
    const names = ['King', 'Rook', 'Bishop', 'Gold', 'Silver', 'Knight', 'Lance', 'Pawn']
    for (const name of names) {
      expect(PATTERNS[name]).toBeDefined()
      expect(PATTERNS[name].base).toBeDefined()
    }
  })

  it('promoted patterns match expectations', () => {
    expect(PATTERNS.King.promoted).toBeNull()
    expect(PATTERNS.Gold.promoted).toBeNull()
    expect(PATTERNS.Rook.promoted).toBe(DRAGON)
    expect(PATTERNS.Bishop.promoted).toBe(HORSE)
    // Silver, Knight, Lance, Pawn all promote to Gold
    expect(PATTERNS.Silver.promoted).toBe(GOLD)
    expect(PATTERNS.Knight.promoted).toBe(GOLD)
    expect(PATTERNS.Lance.promoted).toBe(GOLD)
    expect(PATTERNS.Pawn.promoted).toBe(GOLD)
  })

  it('only uses valid cell values', () => {
    const allGrids = [KING, ROOK, BISHOP, DRAGON, HORSE, GOLD, SILVER, KNIGHT, LANCE, PAWN]
    for (const g of allGrids) {
      for (const row of g) {
        for (const cell of row) {
          expect([null, 'step', 'slide', 'jump']).toContain(cell)
        }
      }
    }
  })
})
