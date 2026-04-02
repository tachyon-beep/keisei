/**
 * Parse a move history JSON string into an array of move objects.
 * Returns empty array on parse failure.
 *
 * @param {string|Array} moveHistoryJson
 * @returns {Array}
 */
export function parseMoves(moveHistoryJson) {
  try {
    return typeof moveHistoryJson === 'string'
      ? JSON.parse(moveHistoryJson)
      : (moveHistoryJson || [])
  } catch {
    return []
  }
}

/**
 * Build paired rows for display: each row has a move number,
 * black's move, white's move, and whether it's the latest row.
 *
 * @param {Array} moves - Array of move objects with .notation
 * @returns {Array<{ num: number, black: string, white: string, isLatest: boolean }>}
 */
export function buildMoveRows(moves) {
  const result = []
  for (let i = 0; i < moves.length; i += 2) {
    result.push({
      num: Math.floor(i / 2) + 1,
      black: moves[i]?.notation || '',
      white: moves[i + 1]?.notation || '',
      isLatest: i >= moves.length - 2,
    })
  }
  return result
}
