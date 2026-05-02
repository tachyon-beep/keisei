import { describe, it, expect } from 'vitest'
import { parseUsi } from './usiCoords.js'

describe('parseUsi', () => {
  // Coordinate convention check:
  //   "9a" -> file=9, rank=a -> col = 9-9 = 0, row = 0 -> idx 0 (top-left)
  //   "1i" -> file=1, rank=i -> col = 9-1 = 8, row = 8 -> idx 80 (bottom-right)
  //   "5e" -> file=5, rank=e -> col = 9-5 = 4, row = 4 -> idx 40 (centre)

  it('parses a simple board move', () => {
    // 7g -> col=2, row=6 -> idx 6*9+2 = 56
    // 7f -> col=2, row=5 -> idx 5*9+2 = 47
    expect(parseUsi('7g7f')).toEqual({
      fromIdx: 56, toIdx: 47, isDrop: false, dropPiece: null,
    })
  })

  it('parses a board move with promotion suffix', () => {
    expect(parseUsi('8h2b+')).toEqual({
      fromIdx: 64, toIdx: 16, isDrop: false, dropPiece: null,
    })
  })

  it('parses a drop', () => {
    // P*5e -> drop pawn at 5e -> idx 40
    expect(parseUsi('P*5e')).toEqual({
      fromIdx: null, toIdx: 40, isDrop: true, dropPiece: 'P',
    })
  })

  it('parses corner squares', () => {
    expect(parseUsi('9a1i')).toEqual({
      fromIdx: 0, toIdx: 80, isDrop: false, dropPiece: null,
    })
  })

  it('returns null for malformed input', () => {
    expect(parseUsi('')).toBeNull()
    expect(parseUsi('garbage')).toBeNull()
    expect(parseUsi('zz')).toBeNull()
    expect(parseUsi(null)).toBeNull()
    expect(parseUsi(undefined)).toBeNull()
  })
})
