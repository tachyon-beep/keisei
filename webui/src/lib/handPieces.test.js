import { describe, it, expect } from 'vitest'
import { getHandPieces } from './handPieces.js'

describe('getHandPieces()', () => {
  it('returns empty array for empty hand', () => {
    expect(getHandPieces({})).toEqual([])
  })

  it('returns single entry for one piece type with count 1', () => {
    const result = getHandPieces({ gold: 1 })
    expect(result).toEqual([
      { type: 'gold', kanji: '金', count: 1 },
    ])
  })

  it('returns multiple entries with correct counts', () => {
    const result = getHandPieces({ pawn: 3, rook: 1, silver: 2 })
    expect(result).toEqual([
      { type: 'rook', kanji: '飛', count: 1 },
      { type: 'silver', kanji: '銀', count: 2 },
      { type: 'pawn', kanji: '歩', count: 3 },
    ])
  })

  it('excludes pieces with count 0', () => {
    const result = getHandPieces({ pawn: 0, gold: 1 })
    expect(result).toEqual([
      { type: 'gold', kanji: '金', count: 1 },
    ])
  })

  it('treats missing keys as 0 (excluded)', () => {
    const result = getHandPieces({ bishop: 2 })
    expect(result).toHaveLength(1)
    expect(result[0].type).toBe('bishop')
  })

  it('follows HAND_PIECE_ORDER: rook first, pawn last', () => {
    const result = getHandPieces({ pawn: 1, rook: 1, lance: 1 })
    expect(result.map(p => p.type)).toEqual(['rook', 'lance', 'pawn'])
  })

  it('uses base kanji, not promoted kanji', () => {
    const result = getHandPieces({ pawn: 1, rook: 1, bishop: 1 })
    expect(result.map(p => p.kanji)).toEqual(['飛', '角', '歩'])
  })

  it('handles all 7 piece types populated', () => {
    const hand = { rook: 1, bishop: 1, gold: 2, silver: 2, knight: 2, lance: 2, pawn: 9 }
    const result = getHandPieces(hand)
    expect(result).toHaveLength(7)
    expect(result.map(p => p.type)).toEqual([
      'rook', 'bishop', 'gold', 'silver', 'knight', 'lance', 'pawn',
    ])
    expect(result.map(p => p.count)).toEqual([1, 1, 2, 2, 2, 2, 9])
  })
})
