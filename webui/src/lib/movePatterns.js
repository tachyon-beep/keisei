/**
 * Movement patterns for Shogi pieces.
 *
 * Each pattern is a 2D grid (3x3 or 5x5 for Knight).
 * Values: null = empty, 'step' = one square (■), 'slide' = unlimited range (arrow).
 * Centre cell is the piece position.
 *
 * Orientation: row 0 = forward (toward opponent), row 2 = backward.
 */

const _ = null
const s = 'step'
const S = 'slide'

// 3x3 patterns — centre is [1][1]
export const KING = [
  [s, s, s],
  [s, _, s],
  [s, s, s],
]

export const ROOK = [
  [_, S, _],
  [S, _, S],
  [_, S, _],
]

export const BISHOP = [
  [S, _, S],
  [_, _, _],
  [S, _, S],
]

// Dragon (promoted Rook) = Rook + 1 step diagonal
export const DRAGON = [
  [s, S, s],
  [S, _, S],
  [s, S, s],
]

// Horse (promoted Bishop) = Bishop + 1 step orthogonal
export const HORSE = [
  [S, s, S],
  [s, _, s],
  [S, s, S],
]

// Gold: all adjacent except back-diagonals
export const GOLD = [
  [s, s, s],
  [s, _, s],
  [_, s, _],
]

// Silver: forward 3 + back diagonals
export const SILVER = [
  [s, s, s],
  [_, _, _],
  [s, _, s],
]

// Lance: slides forward only
export const LANCE = [
  [_, S, _],
  [_, _, _],
  [_, _, _],
]

// Pawn: one step forward
export const PAWN = [
  [_, s, _],
  [_, _, _],
  [_, _, _],
]

// Knight: jumps 2 forward + 1 sideways
// Base 3x3 is all empty; extra row above holds the two jump targets
export const KNIGHT = [
  [_, _, _],
  [_, _, _],
  [_, _, _],
]

// Extra row rendered above the 3x3 grid for the knight
export const KNIGHT_EXTRA = [s, _, s]

/** Map from piece name to { base, promoted } patterns */
/** Map from piece name to { base, promoted, extra? } patterns */
export const PATTERNS = {
  King:   { base: KING,   promoted: null },
  Rook:   { base: ROOK,   promoted: DRAGON },
  Bishop: { base: BISHOP, promoted: HORSE },
  Gold:   { base: GOLD,   promoted: null },
  Silver: { base: SILVER, promoted: GOLD },
  Knight: { base: KNIGHT, promoted: GOLD, extra: KNIGHT_EXTRA },
  Lance:  { base: LANCE,  promoted: GOLD },
  Pawn:   { base: PAWN,   promoted: GOLD },
}
