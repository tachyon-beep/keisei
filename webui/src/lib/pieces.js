/**
 * Shogi piece kanji mappings and rendering helpers.
 */

export const PIECE_KANJI = {
  pawn:   { base: '歩', promoted: 'と' },
  lance:  { base: '香', promoted: '杏' },
  knight: { base: '桂', promoted: '圭' },
  silver: { base: '銀', promoted: '全' },
  gold:   { base: '金', promoted: null },
  bishop: { base: '角', promoted: '馬' },
  rook:   { base: '飛', promoted: '龍' },
  king:   { base: '玉', promoted: null },
}

export const KING_KANJI = {
  black: '王',
  white: '玉',
}

export const HAND_PIECE_ORDER = ['rook', 'bishop', 'gold', 'silver', 'knight', 'lance', 'pawn']

export function pieceKanji(type, promoted, color) {
  if (type === 'king') return KING_KANJI[color] || '玉'
  const entry = PIECE_KANJI[type]
  if (!entry) return '?'
  if (promoted && entry.promoted) return entry.promoted
  return entry.base
}
