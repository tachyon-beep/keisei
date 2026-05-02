/**
 * Parse a USI move string and return board indices (0-80, row*9+col).
 *
 * USI conventions (matches Rust shogi-gym/src/spectator_data.rs#square_notation):
 *   File: digit '1'-'9', counted right-to-left from black's POV. col = 9 - file.
 *   Rank: letter 'a'-'i', top-to-bottom. row = rank.charCodeAt - 'a'.charCodeAt.
 *   Board move: "<from-file><from-rank><to-file><to-rank>[+]"  e.g. "7g7f", "8h2b+"
 *   Drop:       "<piece-char>*<to-file><to-rank>"               e.g. "P*5e"
 *
 * @param {string} usi
 * @returns {{fromIdx: number|null, toIdx: number, isDrop: boolean, dropPiece: string|null}|null}
 */
export function parseUsi(usi) {
  if (typeof usi !== 'string' || usi.length < 2) return null

  // Drop: "P*5e"
  if (usi[1] === '*') {
    if (usi.length < 4) return null
    const piece = usi[0]
    const toIdx = squareToIdx(usi.slice(2, 4))
    if (toIdx === null) return null
    return { fromIdx: null, toIdx, isDrop: true, dropPiece: piece }
  }

  // Board move: "7g7f" or "7g7f+"
  if (usi.length < 4) return null
  const fromIdx = squareToIdx(usi.slice(0, 2))
  const toIdx = squareToIdx(usi.slice(2, 4))
  if (fromIdx === null || toIdx === null) return null
  return { fromIdx, toIdx, isDrop: false, dropPiece: null }
}

function squareToIdx(sq) {
  if (sq.length !== 2) return null
  const file = sq.charCodeAt(0) - '1'.charCodeAt(0) + 1   // '1'..'9' -> 1..9
  const rank = sq.charCodeAt(1) - 'a'.charCodeAt(0)        // 'a'..'i' -> 0..8
  if (file < 1 || file > 9 || rank < 0 || rank > 8) return null
  const col = 9 - file
  return rank * 9 + col
}
