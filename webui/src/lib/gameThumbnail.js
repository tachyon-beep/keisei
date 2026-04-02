import { safeParse } from './safeParse.js'

/**
 * Parse a game's board_json field into a board array.
 * Uses safeParse to deduplicate the inline try/catch.
 */
export function parseBoard(game) {
  if (game.board_json == null) return game.board || []
  return safeParse(game.board_json, game.board || []) || []
}

/**
 * Build the status text for a game thumbnail.
 */
export function getStatusText(game) {
  return game.is_over
    ? (game.result || '').replaceAll('_', ' ')
    : `Ply ${game.ply}`
}

/**
 * Compute advantage and confidence for a game.
 * Returns { blackAdv, confident, favours }.
 */
export function getAdvantage(game) {
  const value = game.value_estimate || 0
  const currentPlayer = game.current_player || 'black'
  const blackAdv = currentPlayer === 'black' ? value : -value
  const confident = Math.abs(blackAdv) > 0.3
  const favours = blackAdv > 0 ? 'black' : 'white'
  return { blackAdv, confident, favours }
}
