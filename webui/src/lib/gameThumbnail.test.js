import { describe, it, expect } from 'vitest'
import { parseBoard, getStatusText, getAdvantage } from './gameThumbnail.js'

describe('parseBoard', () => {
  it('parses valid JSON string to array', () => {
    const game = { board_json: '[1,2,3]' }
    expect(parseBoard(game)).toEqual([1, 2, 3])
  })

  it('returns empty array for invalid JSON string', () => {
    const game = { board_json: '{broken' }
    expect(parseBoard(game)).toEqual([])
  })

  it('passes through non-string board_json (object)', () => {
    const board = [{ type: 'pawn', color: 'black' }]
    const game = { board_json: board }
    expect(parseBoard(game)).toBe(board)
  })

  it('uses game.board when board_json is missing', () => {
    const board = [{ type: 'king', color: 'white' }]
    const game = { board }
    expect(parseBoard(game)).toBe(board)
  })

  it('returns empty array when both board_json and board are missing', () => {
    expect(parseBoard({})).toEqual([])
  })

  it('returns empty array for null board_json', () => {
    const game = { board_json: null }
    expect(parseBoard(game)).toEqual([])
  })
})

describe('getStatusText', () => {
  it('replaces underscores with spaces when game is over', () => {
    const game = { is_over: true, result: 'black_wins_by_checkmate' }
    expect(getStatusText(game)).toBe('black wins by checkmate')
  })

  it('returns empty string when game is over with no result', () => {
    const game = { is_over: true }
    expect(getStatusText(game)).toBe('')
  })

  it('shows ply count when game is not over', () => {
    const game = { is_over: false, ply: 42 }
    expect(getStatusText(game)).toBe('Ply 42')
  })
})

describe('getAdvantage', () => {
  it('black player with positive value -> positive blackAdv, favours black', () => {
    const game = { value_estimate: 0.8, current_player: 'black' }
    const result = getAdvantage(game)
    expect(result.blackAdv).toBe(0.8)
    expect(result.favours).toBe('black')
  })

  it('black player with negative value -> negative blackAdv, favours white', () => {
    const game = { value_estimate: -0.5, current_player: 'black' }
    const result = getAdvantage(game)
    expect(result.blackAdv).toBe(-0.5)
    expect(result.favours).toBe('white')
  })

  it('white player with positive value -> negative blackAdv (sign flip)', () => {
    const game = { value_estimate: 0.6, current_player: 'white' }
    const result = getAdvantage(game)
    expect(result.blackAdv).toBe(-0.6)
    expect(result.favours).toBe('white')
  })

  it('white player with negative value -> positive blackAdv (sign flip)', () => {
    const game = { value_estimate: -0.7, current_player: 'white' }
    const result = getAdvantage(game)
    expect(result.blackAdv).toBe(0.7)
    expect(result.favours).toBe('black')
  })

  it('confident when |blackAdv| > 0.3', () => {
    const game = { value_estimate: 0.5, current_player: 'black' }
    expect(getAdvantage(game).confident).toBe(true)
  })

  it('not confident when |blackAdv| <= 0.3', () => {
    const game = { value_estimate: 0.3, current_player: 'black' }
    expect(getAdvantage(game).confident).toBe(false)
  })

  it('defaults value_estimate to 0 (not confident)', () => {
    const game = { current_player: 'black' }
    const result = getAdvantage(game)
    expect(result.blackAdv).toBe(0)
    expect(result.confident).toBe(false)
  })

  it('defaults current_player to black', () => {
    const game = { value_estimate: 0.5 }
    const result = getAdvantage(game)
    expect(result.blackAdv).toBe(0.5)
    expect(result.favours).toBe('black')
  })
})
