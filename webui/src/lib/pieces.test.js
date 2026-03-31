import { describe, it, expect } from 'vitest'
import { PIECE_KANJI, KING_KANJI, HAND_PIECE_ORDER, pieceKanji } from './pieces.js'

describe('PIECE_KANJI', () => {
  it('has all 8 piece types', () => {
    const types = Object.keys(PIECE_KANJI)
    expect(types).toEqual(['pawn', 'lance', 'knight', 'silver', 'gold', 'bishop', 'rook', 'king'])
  })

  it('promotable pieces have promoted kanji', () => {
    for (const type of ['pawn', 'lance', 'knight', 'silver', 'bishop', 'rook']) {
      expect(PIECE_KANJI[type].promoted).toBeTruthy()
    }
  })

  it('gold and king cannot promote', () => {
    expect(PIECE_KANJI.gold.promoted).toBeNull()
    expect(PIECE_KANJI.king.promoted).toBeNull()
  })
})

describe('KING_KANJI', () => {
  it('distinguishes black and white king', () => {
    expect(KING_KANJI.black).toBe('王')
    expect(KING_KANJI.white).toBe('玉')
  })
})

describe('HAND_PIECE_ORDER', () => {
  it('has 7 droppable piece types (no king)', () => {
    expect(HAND_PIECE_ORDER).toHaveLength(7)
    expect(HAND_PIECE_ORDER).not.toContain('king')
  })

  it('orders rook first, pawn last (descending value)', () => {
    expect(HAND_PIECE_ORDER[0]).toBe('rook')
    expect(HAND_PIECE_ORDER[HAND_PIECE_ORDER.length - 1]).toBe('pawn')
  })
})

describe('pieceKanji()', () => {
  it('returns base kanji for unpromoted pieces', () => {
    expect(pieceKanji('pawn', false, 'black')).toBe('歩')
    expect(pieceKanji('rook', false, 'black')).toBe('飛')
    expect(pieceKanji('bishop', false, 'white')).toBe('角')
  })

  it('returns promoted kanji for promoted pieces', () => {
    expect(pieceKanji('pawn', true, 'black')).toBe('と')
    expect(pieceKanji('rook', true, 'black')).toBe('龍')
    expect(pieceKanji('silver', true, 'white')).toBe('全')
  })

  it('returns base kanji when promoted=true but piece cannot promote', () => {
    expect(pieceKanji('gold', true, 'black')).toBe('金')
  })

  it('returns correct king kanji per color', () => {
    expect(pieceKanji('king', false, 'black')).toBe('王')
    expect(pieceKanji('king', false, 'white')).toBe('玉')
  })

  it('returns fallback for king with unknown color', () => {
    expect(pieceKanji('king', false, 'red')).toBe('玉')
  })

  it('returns ? for unknown piece type', () => {
    expect(pieceKanji('dragon', false, 'black')).toBe('?')
  })
})
