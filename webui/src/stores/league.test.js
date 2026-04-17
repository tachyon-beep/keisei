import { describe, it, expect, beforeEach, vi } from 'vitest'
import { get } from 'svelte/store'
import {
  leagueEntries, leagueResults, eloHistory,
  leagueRanked, entryWLD, headToHead, eloDelta, leagueStats,
  historicalLibrary, gauntletResults, leagueTransitions,
  learnerEntry, leagueByRole, transitionCounts, headToHeadRaw,
} from './league.js'
import { trainingState } from './training.js'

beforeEach(() => {
  leagueEntries.set([])
  leagueResults.set([])
  eloHistory.set([])
  historicalLibrary.set([])
  gauntletResults.set([])
  leagueTransitions.set([])
  headToHeadRaw.set([])
  trainingState.set(null)
})

describe('leagueRanked', () => {
  it('returns empty array when no entries', () => {
    expect(get(leagueRanked)).toEqual([])
  })

  it('sorts entries by elo_rating descending and injects rank', () => {
    leagueEntries.set([
      { id: 1, architecture: 'a', elo_rating: 900, games_played: 10, created_epoch: 1, status: 'active' },
      { id: 2, architecture: 'b', elo_rating: 1200, games_played: 20, created_epoch: 2, status: 'active' },
      { id: 3, architecture: 'c', elo_rating: 1100, games_played: 15, created_epoch: 3, status: 'active' },
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
    // Clear persisted state — league.js persists events to localStorage,
    // and jsdom shares it across vi.resetModules() calls.
    try { localStorage.removeItem('keisei_league_events') } catch {}
    vi.resetModules()
    const mod = await import('./league.js')
    diffLeagueEntries = mod.diffLeagueEntries
    leagueEvents = mod.leagueEvents
    // resetLeagueEvents clears _prevEntryMap, _prevRanks, and the store
    mod.resetLeagueEvents()
    mod.leagueEntries.set([])
    mod.leagueResults.set([])
    mod.eloHistory.set([])
  })

  const entry = (id, elo, name) => ({
    id, elo_rating: elo, display_name: name, architecture: `arch-${id}`, status: 'active',
  })

  it('first call seeds state without generating events (avoids phantom arrivals on refresh)', () => {
    diffLeagueEntries([entry(1, 1200, 'A'), entry(2, 1000, 'B')])
    const events = get(leagueEvents)
    // First call seeds _prevEntryMap and returns early — no events generated
    expect(events).toHaveLength(0)
  })

  it('clears stale persisted events on first call even when entries exist (run restart)', () => {
    // Simulate stale events left over from a previous training run
    leagueEvents.set([{ time: '04:06:26', type: 'departure', icon: '←', name: 'Stale-Bot', detail: 'retired' }])
    // First diff call should always clear stale events, regardless of entry count
    diffLeagueEntries([entry(1, 1200, 'A'), entry(2, 1000, 'B')])
    const events = get(leagueEvents)
    expect(events).toHaveLength(0)
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
    const e = { id: 1, elo_rating: 1000, architecture: 'resnet-v2', status: 'active' }
    // Seed with a different entry first (first call seeds state, no events)
    diffLeagueEntries([entry(99, 800, 'Seed')])
    leagueEvents.set([])
    // Now add the nameless entry — should detect arrival with arch fallback
    diffLeagueEntries([entry(99, 800, 'Seed'), e])
    const events = get(leagueEvents)
    const arrival = events.find(ev => ev.type === 'arrival')
    expect(arrival).toBeDefined()
    expect(arrival.name).toBe('resnet-v2')
  })
})

// --- Derived stores: no module-level mutable state, standard imports work fine.

describe('entryWLD', () => {
  it('returns empty map when no results', () => {
    expect(get(entryWLD).size).toBe(0)
  })

  it('aggregates W/L/D for A and mirrors for B', () => {
    leagueResults.set([
      { entry_a_id: 1, entry_b_id: 2, wins_a: 3, wins_b: 1, draws: 2 },
    ])
    const wld = get(entryWLD)
    // Side A
    expect(wld.get(1)).toEqual({ w: 3, l: 1, d: 2 })
    // Side B (mirrored)
    expect(wld.get(2)).toEqual({ w: 1, l: 3, d: 2 })
  })

  it('accumulates across multiple result rows', () => {
    leagueResults.set([
      { entry_a_id: 1, entry_b_id: 2, wins_a: 2, wins_b: 0, draws: 0 },
      { entry_a_id: 1, entry_b_id: 3, wins_a: 1, wins_b: 1, draws: 1 },
    ])
    const wld = get(entryWLD)
    expect(wld.get(1)).toEqual({ w: 3, l: 1, d: 1 })
  })

  it('handles zero/missing wins_a/wins_b/draws', () => {
    leagueResults.set([
      { entry_a_id: 1, entry_b_id: 2 }, // all undefined
    ])
    const wld = get(entryWLD)
    expect(wld.get(1)).toEqual({ w: 0, l: 0, d: 0 })
  })
})

describe('headToHead', () => {
  it('returns empty map when no h2h data', () => {
    expect(get(headToHead).size).toBe(0)
  })

  it('computes bidirectional win rates from pre-aggregated data', () => {
    // Backend sends canonical ordering (entry_a_id < entry_b_id)
    headToHeadRaw.set([
      { entry_a_id: 1, entry_b_id: 2, wins_a: 3, wins_b: 1, draws: 0, games: 4 },
    ])
    const h2h = get(headToHead)
    const ab = h2h.get('1-2')
    expect(ab.w).toBe(3)
    expect(ab.l).toBe(1)
    expect(ab.winRate).toBe(0.75)
    const ba = h2h.get('2-1')
    expect(ba.w).toBe(1)
    expect(ba.l).toBe(3)
    expect(ba.winRate).toBe(0.25)
  })

  it('returns null winRate when total is 0', () => {
    headToHeadRaw.set([
      { entry_a_id: 1, entry_b_id: 2, wins_a: 0, wins_b: 0, draws: 0, games: 0 },
    ])
    const h2h = get(headToHead)
    expect(h2h.get('1-2').winRate).toBeNull()
  })

  it('includes draws in total', () => {
    headToHeadRaw.set([
      { entry_a_id: 1, entry_b_id: 2, wins_a: 1, wins_b: 1, draws: 2, games: 4 },
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
    leagueEntries.set([{ id: 1, elo_rating: 1000, architecture: 'a', status: 'active' }])
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
      { id: 1, elo_rating: 900, architecture: 'a', status: 'active' },
      { id: 2, elo_rating: 1200, architecture: 'b', status: 'active' },
      { id: 3, elo_rating: 1100, architecture: 'c', status: 'active' },
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

describe('learnerEntry', () => {
  it('returns null when trainingState is null', () => {
    expect(get(learnerEntry)).toBeNull()
  })

  it('returns null when trainingState has no display_name key', () => {
    trainingState.set({})
    leagueEntries.set([
      { id: 1, display_name: 'Bot-A', elo_rating: 1000, status: 'active' },
    ])
    expect(get(learnerEntry)).toBeNull()
  })

  it('returns null when display_name is set but no entry matches', () => {
    trainingState.set({ display_name: 'NonExistent' })
    leagueEntries.set([
      { id: 1, display_name: 'Bot-A', elo_rating: 1000, status: 'active' },
    ])
    expect(get(learnerEntry)).toBeNull()
  })

  it('returns matching entry when display_name matches', () => {
    trainingState.set({ display_name: 'Bot-A' })
    leagueEntries.set([
      { id: 1, display_name: 'Bot-A', elo_rating: 1000, status: 'active' },
      { id: 2, display_name: 'Bot-B', elo_rating: 1100, status: 'active' },
    ])
    const entry = get(learnerEntry)
    expect(entry).not.toBeNull()
    expect(entry.id).toBe(1)
    expect(entry.display_name).toBe('Bot-A')
  })
})

describe('leagueByRole', () => {
  it('returns empty map when no entries', () => {
    expect(get(leagueByRole).size).toBe(0)
  })

  it('groups active entries by role', () => {
    leagueEntries.set([
      { id: 1, role: 'frontier_static', elo_rating: 1200, status: 'active' },
      { id: 2, role: 'frontier_static', elo_rating: 1100, status: 'active' },
      { id: 3, role: 'dynamic', elo_rating: 1000, status: 'active' },
    ])
    const byRole = get(leagueByRole)
    expect(byRole.get('frontier_static')).toHaveLength(2)
    expect(byRole.get('dynamic')).toHaveLength(1)
  })

  it('places entries with null/unknown role in "other" bucket', () => {
    leagueEntries.set([
      { id: 1, role: null, elo_rating: 1000, status: 'active' },
      { id: 2, role: 'some_future_role', elo_rating: 900, status: 'active' },
    ])
    const byRole = get(leagueByRole)
    expect(byRole.get('other')).toHaveLength(2)
  })

  it('excludes retired entries (inherited from leagueRanked)', () => {
    leagueEntries.set([
      { id: 1, role: 'dynamic', elo_rating: 1000, status: 'active' },
      { id: 2, role: 'dynamic', elo_rating: 900, status: 'retired' },
    ])
    const byRole = get(leagueByRole)
    expect(byRole.get('dynamic')).toHaveLength(1)
  })
})

describe('transitionCounts', () => {
  it('returns zeros when no transitions', () => {
    const counts = get(transitionCounts)
    expect(counts).toEqual({ promotions: 0, evictions: 0, admissions: 0 })
  })

  it('counts admissions (null from_status to active)', () => {
    leagueTransitions.set([
      { id: 1, from_status: null, to_status: 'active', from_role: null, to_role: 'dynamic' },
    ])
    expect(get(transitionCounts).admissions).toBe(1)
  })

  it('counts admissions (from_status key absent)', () => {
    leagueTransitions.set([
      { id: 1, to_status: 'active', from_role: null, to_role: 'dynamic' },
    ])
    expect(get(transitionCounts).admissions).toBe(1)
  })

  it('counts evictions (active to retired)', () => {
    leagueTransitions.set([
      { id: 1, from_status: 'active', to_status: 'retired', from_role: 'dynamic', to_role: 'dynamic' },
    ])
    expect(get(transitionCounts).evictions).toBe(1)
  })

  it('counts promotions (role upgrade while active)', () => {
    leagueTransitions.set([
      { id: 1, from_status: 'active', to_status: 'active', from_role: 'dynamic', to_role: 'frontier_static' },
      { id: 2, from_status: 'active', to_status: 'active', from_role: 'recent_fixed', to_role: 'frontier_static' },
      { id: 3, from_status: 'active', to_status: 'active', from_role: 'dynamic', to_role: 'recent_fixed' },
    ])
    expect(get(transitionCounts).promotions).toBe(3)
  })

  it('ignores unrecognised transition types', () => {
    leagueTransitions.set([
      { id: 1, from_status: 'active', to_status: 'active', from_role: 'dynamic', to_role: 'dynamic' },
      { id: 2, from_status: 'retired', to_status: 'active', from_role: 'dynamic', to_role: 'dynamic' },
    ])
    const counts = get(transitionCounts)
    expect(counts).toEqual({ promotions: 0, evictions: 0, admissions: 0 })
  })

  it('status transitions take precedence over role transitions', () => {
    leagueTransitions.set([
      { id: 1, from_status: 'active', to_status: 'retired', from_role: 'dynamic', to_role: 'frontier_static' },
    ])
    const counts = get(transitionCounts)
    expect(counts.evictions).toBe(1)
    expect(counts.promotions).toBe(0)
  })
})
