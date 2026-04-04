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
 * @param {Array} moves - Array of move objects with .notation and .usi
 * @param {'western'|'japanese'|'usi'} style - Notation style
 * @returns {Array<{ num: number, black: string, white: string, isLatest: boolean }>}
 */
export function buildMoveRows(moves, style = 'western') {
  const getText = style === 'usi'
    ? (m) => m?.usi || m?.notation || ''
    : style === 'japanese'
      ? (m) => toJapanese(m?.notation || '')
      : (m) => m?.notation || ''
  const result = []
  for (let i = 0; i < moves.length; i += 2) {
    result.push({
      num: Math.floor(i / 2) + 1,
      black: getText(moves[i]),
      white: getText(moves[i + 1]),
      isLatest: i >= moves.length - 2,
    })
  }
  return result
}

const RANK_KANJI = { a: '一', b: '二', c: '三', d: '四', e: '五', f: '六', g: '七', h: '八', i: '九' }
const FILE_FULL = { '1': '１', '2': '２', '3': '３', '4': '４', '5': '５', '6': '６', '7': '７', '8': '８', '9': '９' }

/**
 * Convert Hodges notation coordinates to Japanese: "P-7f" → "P-７六"
 */
export function toJapanese(notation) {
  if (!notation) return ''
  return notation.replace(/([1-9])([a-i])/g, (_, file, rank) => {
    return (FILE_FULL[file] || file) + (RANK_KANJI[rank] || rank)
  })
}
