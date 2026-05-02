/**
 * Movement patterns for Shogi pieces.
 *
 * Each pattern is a 3x3 grid. Centre cell [1][1] is the piece position.
 * Values:
 *   null   = no move
 *   'step' = one square (■)
 *   'slide'= unlimited range in that direction (arrow)
 *   'jump' = Knight leap (⇖/⇗) — drawn in a corner of row 0 to keep all
 *            piece legends visually 3 rows tall, even though the actual
 *            landing square is 2 forward + 1 sideways.
 *
 * Orientation: row 0 = forward (toward opponent), row 2 = backward.
 */

const _ = null
const s = 'step'
const S = 'slide'
const J = 'jump'

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

// Knight: jumps 2 forward + 1 sideways. Drawn as 'jump' markers in the
// forward corners so the legend stays a uniform 3 rows.
export const KNIGHT = [
  [J, _, J],
  [_, _, _],
  [_, _, _],
]

/** Map from piece name to { base, promoted } patterns */
export const PATTERNS = {
  King:   { base: KING,   promoted: null },
  Rook:   { base: ROOK,   promoted: DRAGON },
  Bishop: { base: BISHOP, promoted: HORSE },
  Gold:   { base: GOLD,   promoted: null },
  Silver: { base: SILVER, promoted: GOLD },
  Knight: { base: KNIGHT, promoted: GOLD },
  Lance:  { base: LANCE,  promoted: GOLD },
  Pawn:   { base: PAWN,   promoted: GOLD },
}
