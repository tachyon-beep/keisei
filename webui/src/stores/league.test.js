import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { leagueEntries, leagueResults, eloHistory, leagueRanked } from './league.js'

beforeEach(() => {
  leagueEntries.set([])
  leagueResults.set([])
  eloHistory.set([])
})

describe('leagueRanked', () => {
  it('returns empty array when no entries', () => {
    expect(get(leagueRanked)).toEqual([])
  })

  it('sorts entries by elo_rating descending and injects rank', () => {
    leagueEntries.set([
      { id: 1, architecture: 'a', elo_rating: 900, games_played: 10, created_epoch: 1 },
      { id: 2, architecture: 'b', elo_rating: 1200, games_played: 20, created_epoch: 2 },
      { id: 3, architecture: 'c', elo_rating: 1100, games_played: 15, created_epoch: 3 },
    ])
    const ranked = get(leagueRanked)
    expect(ranked[0]).toEqual(expect.objectContaining({ id: 2, rank: 1, elo_rating: 1200 }))
    expect(ranked[1]).toEqual(expect.objectContaining({ id: 3, rank: 2, elo_rating: 1100 }))
    expect(ranked[2]).toEqual(expect.objectContaining({ id: 1, rank: 3, elo_rating: 900 }))
  })
})
