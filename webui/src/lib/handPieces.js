import { PIECE_KANJI, HAND_PIECE_ORDER } from './pieces.js'

/**
 * Build display-ready hand pieces from a hand object.
 * @param {object} hand - Map of piece type to count (e.g., { pawn: 3, gold: 1 })
 * @returns {Array<{type: string, kanji: string, count: number}>}
 */
export function getHandPieces(hand) {
  return HAND_PIECE_ORDER
    .filter(type => (hand[type] || 0) > 0)
    .map(type => ({
      type,
      kanji: PIECE_KANJI[type].base,
      count: hand[type],
    }))
}
