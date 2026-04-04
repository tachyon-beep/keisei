import { describe, it, expect, beforeEach, vi } from 'vitest'
import { get } from 'svelte/store'
import {
  leagueEntries, leagueResults, eloHistory,
  leagueRanked, entryWLD, headToHead, eloDelta, leagueStats,
} from './league.js'

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

// --- diffLeagueEntries uses module-level mutable state (_prevEntryMap, _prevRanks).
// We must use vi.resetModules() + dynamic import to get a fresh module per test.

describe('diffLeagueEntries', () => {
  let diffLeagueEntries, leagueEvents

  beforeEach(async () => {
    vi.resetModules()
    const mod = await import('./league.js')
    diffLeagueEntries = mod.diffLeagueEntries
    leagueEvents = mod.leagueEvents
    mod.leagueEntries.set([])
    mod.leagueResults.set([])
    mod.eloHistory.set([])
    leagueEvents.set([])
  })

  const entry = (id, elo, name) => ({
    id, elo_rating: elo, display_name: name, architecture: `arch-${id}`,
  })

  it('first call generates arrival events but no rank-change events', () => {
    diffLeagueEntries([entry(1, 1200, 'A'), entry(2, 1000, 'B')])
    const events = get(leagueEvents)
    // Every entry looks "new" on the first call (empty _prevEntryMap)
    expect(events).toHaveLength(2)
    expect(events.every(e => e.type === 'arrival')).toBe(true)
    // No rank changes on first load
    expect(events.filter(e => e.type === 'promotion' || e.type === 'demotion')).toHaveLength(0)
  })

  it('detects an arrival on subsequent calls', () => {
    diffLeagueEntries([entry(1, 1200, 'A')])
    leagueEvents.set([]) // clear first-call arrivals
    diffLeagueEntries([entry(1, 1200, 'A'), entry(2, 1000, 'B')])
    const events = get(leagueEvents)
    expect(events).toHaveLength(1)
    expect(events[0].type).toBe('arrival')
    expect(events[0].name).toBe('B')
  })

  it('detects a departure', () => {
    diffLeagueEntries([entry(1, 1200, 'A'), entry(2, 1000, 'B')])
    leagueEvents.set([]) // clear first-call arrivals
    diffLeagueEntries([entry(1, 1200, 'A')])
    const events = get(leagueEvents)
    expect(events).toHaveLength(1)
    expect(events[0].type).toBe('departure')
    expect(events[0].name).toBe('B')
  })

  it('detects top-3 rank promotion', () => {
    diffLeagueEntries([entry(1, 1200, 'A'), entry(2, 1000, 'B'), entry(3, 800, 'C'), entry(4, 600, 'D')])
    diffLeagueEntries([entry(1, 1200, 'A'), entry(2, 1000, 'B'), entry(3, 800, 'C'), entry(4, 1500, 'D')])
    const events = get(leagueEvents)
    const promo = events.find(e => e.type === 'promotion')
    expect(promo).toBeDefined()
    expect(promo.name).toBe('D')
    expect(promo.detail).toContain('4')
    expect(promo.detail).toContain('1')
  })

  it('detects top-3 rank demotion', () => {
    diffLeagueEntries([entry(1, 1200, 'A'), entry(2, 1000, 'B'), entry(3, 800, 'C'), entry(4, 600, 'D')])
    diffLeagueEntries([entry(1, 500, 'A'), entry(2, 1000, 'B'), entry(3, 800, 'C'), entry(4, 600, 'D')])
    const events = get(leagueEvents)
    const demotion = events.find(e => e.type === 'demotion')
    expect(demotion).toBeDefined()
    expect(demotion.name).toBe('A')
  })

  it('suppresses rank changes outside top 3', () => {
    diffLeagueEntries([
      entry(1, 1200, 'A'), entry(2, 1000, 'B'), entry(3, 800, 'C'),
      entry(4, 600, 'D'), entry(5, 400, 'E'),
    ])
    diffLeagueEntries([
      entry(1, 1200, 'A'), entry(2, 1000, 'B'), entry(3, 800, 'C'),
      entry(4, 300, 'D'), entry(5, 500, 'E'),
    ])
    const events = get(leagueEvents)
    const rankEvents = events.filter(e => e.type === 'promotion' || e.type === 'demotion')
    expect(rankEvents).toHaveLength(0)
  })

  it('suppresses rank changes when pool size is 1', () => {
    diffLeagueEntries([entry(1, 1200, 'A')])
    leagueEvents.set([]) // clear first-call arrival
    diffLeagueEntries([entry(1, 900, 'A')])
    expect(get(leagueEvents)).toEqual([])
  })

  it('truncates events at MAX_EVENTS (50)', () => {
    diffLeagueEntries([entry(1, 1200, 'A')])
    for (let i = 2; i <= 61; i++) {
      diffLeagueEntries([entry(1, 1200, 'A'), entry(i, 1000 + i, `Bot${i}`)])
    }
    expect(get(leagueEvents).length).toBeLessThanOrEqual(50)
  })

  it('uses architecture as fallback when display_name is missing', () => {
    const e = { id: 1, elo_rating: 1000, architecture: 'resnet-v2' }
    diffLeagueEntries([])
    diffLeagueEntries([e])
    const events = get(leagueEvents)
    expect(events[0].name).toBe('resnet-v2')
  })
})

// --- Derived stores: no module-level mutable state, standard imports work fine.

describe('entryWLD', () => {
  it('returns empty map when no results', () => {
    expect(get(entryWLD).size).toBe(0)
  })

  it('aggregates W/L/D for learner and mirrors for opponent', () => {
    leagueResults.set([
      { learner_id: 1, opponent_id: 2, wins: 3, losses: 1, draws: 2 },
    ])
    const wld = get(entryWLD)
    // Learner side
    expect(wld.get(1)).toEqual({ w: 3, l: 1, d: 2 })
    // Opponent side (mirrored)
    expect(wld.get(2)).toEqual({ w: 1, l: 3, d: 2 })
  })

  it('accumulates across multiple result rows', () => {
    leagueResults.set([
      { learner_id: 1, opponent_id: 2, wins: 2, losses: 0, draws: 0 },
      { learner_id: 1, opponent_id: 3, wins: 1, losses: 1, draws: 1 },
    ])
    const wld = get(entryWLD)
    expect(wld.get(1)).toEqual({ w: 3, l: 1, d: 1 })
  })

  it('handles zero/missing wins/losses/draws', () => {
    leagueResults.set([
      { learner_id: 1, opponent_id: 2 }, // all undefined
    ])
    const wld = get(entryWLD)
    expect(wld.get(1)).toEqual({ w: 0, l: 0, d: 0 })
  })
})

describe('headToHead', () => {
  it('returns empty map when no results', () => {
    expect(get(headToHead).size).toBe(0)
  })

  it('computes bidirectional win rates', () => {
    leagueResults.set([
      { learner_id: 1, opponent_id: 2, wins: 3, losses: 1, draws: 0 },
    ])
    const h2h = get(headToHead)
    const lo = h2h.get('1-2')
    expect(lo.w).toBe(3)
    expect(lo.l).toBe(1)
    expect(lo.winRate).toBe(0.75)
    const ol = h2h.get('2-1')
    expect(ol.w).toBe(1)
    expect(ol.l).toBe(3)
    expect(ol.winRate).toBe(0.25)
  })

  it('returns null winRate when total is 0', () => {
    leagueResults.set([
      { learner_id: 1, opponent_id: 2, wins: 0, losses: 0, draws: 0 },
    ])
    const h2h = get(headToHead)
    expect(h2h.get('1-2').winRate).toBeNull()
  })

  it('includes draws in total', () => {
    leagueResults.set([
      { learner_id: 1, opponent_id: 2, wins: 1, losses: 1, draws: 2 },
    ])
    const h2h = get(headToHead)
    expect(h2h.get('1-2').total).toBe(4)
    expect(h2h.get('1-2').winRate).toBe(0.25)
  })
})

describe('eloDelta', () => {
  it('returns empty map when no history', () => {
    expect(get(eloDelta).size).toBe(0)
  })

  it('computes delta from earliest to latest epoch', () => {
    eloHistory.set([
      { entry_id: 1, epoch: 5, elo_rating: 1050 },
      { entry_id: 1, epoch: 1, elo_rating: 1000 },
      { entry_id: 1, epoch: 10, elo_rating: 1100 },
    ])
    const deltas = get(eloDelta)
    // delta = last(1100) - first(1000) = 100
    expect(deltas.get(1)).toBe(100)
  })

  it('returns 0 delta for single data point', () => {
    eloHistory.set([
      { entry_id: 1, epoch: 5, elo_rating: 1050 },
    ])
    expect(get(eloDelta).get(1)).toBe(0)
  })

  it('handles negative delta (elo decreased)', () => {
    eloHistory.set([
      { entry_id: 1, epoch: 1, elo_rating: 1200 },
      { entry_id: 1, epoch: 10, elo_rating: 900 },
    ])
    expect(get(eloDelta).get(1)).toBe(-300)
  })

  it('tracks multiple entries independently', () => {
    eloHistory.set([
      { entry_id: 1, epoch: 1, elo_rating: 1000 },
      { entry_id: 1, epoch: 5, elo_rating: 1100 },
      { entry_id: 2, epoch: 1, elo_rating: 1200 },
      { entry_id: 2, epoch: 5, elo_rating: 1150 },
    ])
    const deltas = get(eloDelta)
    expect(deltas.get(1)).toBe(100)
    expect(deltas.get(2)).toBe(-50)
  })
})

describe('leagueStats', () => {
  it('returns null when no entries', () => {
    expect(get(leagueStats)).toBeNull()
  })

  it('computes summary for a single entry', () => {
    leagueEntries.set([{ id: 1, elo_rating: 1000, architecture: 'a' }])
    leagueResults.set([])
    const stats = get(leagueStats)
    expect(stats.poolSize).toBe(1)
    expect(stats.totalMatches).toBe(0)
    expect(stats.eloMin).toBe(1000)
    expect(stats.eloMax).toBe(1000)
    expect(stats.eloSpread).toBe(0)
    expect(stats.topEntry.id).toBe(1)
  })

  it('computes spread across multiple entries', () => {
    leagueEntries.set([
      { id: 1, elo_rating: 900, architecture: 'a' },
      { id: 2, elo_rating: 1200, architecture: 'b' },
      { id: 3, elo_rating: 1100, architecture: 'c' },
    ])
    leagueResults.set([{ id: 10 }, { id: 11 }])
    const stats = get(leagueStats)
    expect(stats.poolSize).toBe(3)
    expect(stats.totalMatches).toBe(2)
    expect(stats.eloMin).toBe(900)
    expect(stats.eloMax).toBe(1200)
    expect(stats.eloSpread).toBe(300)
    expect(stats.topEntry.id).toBe(2)
  })
})
