import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { games, selectedGameId, selectedGame, selectedOpponent } from './games.js'
import { leagueEntries } from './league.js'

beforeEach(() => {
  games.set([])
  selectedGameId.set(0)
  leagueEntries.set([])
})

describe('games store', () => {
  it('starts empty', () => {
    expect(get(games)).toEqual([])
  })
})

describe('selectedGame derived store', () => {
  const gameList = [
    { game_id: 0, board: '...', ply: 10 },
    { game_id: 1, board: '...', ply: 20 },
    { game_id: 2, board: '...', ply: 30 },
  ]

  it('returns null when no games exist', () => {
    expect(get(selectedGame)).toBeNull()
  })

  it('selects game by selectedGameId', () => {
    games.set(gameList)
    selectedGameId.set(1)
    expect(get(selectedGame).game_id).toBe(1)
    expect(get(selectedGame).ply).toBe(20)
  })

  it('falls back to first game when selectedGameId has no match', () => {
    games.set(gameList)
    selectedGameId.set(999)
    expect(get(selectedGame).game_id).toBe(0)
  })

  it('returns null when games array is empty and ID is set', () => {
    selectedGameId.set(5)
    expect(get(selectedGame)).toBeNull()
  })

  it('updates reactively when games list changes', () => {
    games.set(gameList)
    selectedGameId.set(2)
    expect(get(selectedGame).ply).toBe(30)

    // Replace game list — game_id 2 now has different data
    games.set([{ game_id: 2, board: 'new', ply: 50 }])
    expect(get(selectedGame).ply).toBe(50)
  })
})

describe('selectedOpponent derived store', () => {
  it('returns null when game has no opponent_id', () => {
    games.set([{ game_id: 0, opponent_id: null }])
    selectedGameId.set(0)
    expect(get(selectedOpponent)).toBeNull()
  })

  it('returns null when no league entries exist', () => {
    games.set([{ game_id: 0, opponent_id: 5 }])
    selectedGameId.set(0)
    expect(get(selectedOpponent)).toBeNull()
  })

  it('returns opponent entry when opponent_id matches a league entry', () => {
    leagueEntries.set([
      { id: 5, architecture: 'transformer_ep00008', elo_rating: 1180, games_played: 124 },
    ])
    games.set([{ game_id: 0, opponent_id: 5 }])
    selectedGameId.set(0)
    const opp = get(selectedOpponent)
    expect(opp).toEqual({
      architecture: 'transformer_ep00008',
      elo_rating: 1180,
      games_played: 124,
    })
  })
})
