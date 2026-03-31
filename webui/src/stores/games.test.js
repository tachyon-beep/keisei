import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { games, selectedGameId, selectedGame } from './games.js'

beforeEach(() => {
  games.set([])
  selectedGameId.set(0)
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
