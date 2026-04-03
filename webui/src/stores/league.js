import { writable, derived } from 'svelte/store'

export const leagueEntries = writable([])
export const leagueResults = writable([])
export const eloHistory = writable([])

/** Aggregate W/L/D totals keyed by entry id */
export const entryWLD = derived(leagueResults, ($results) => {
  const map = new Map()
  for (const r of $results) {
    // Learner side
    const l = map.get(r.learner_id) || { w: 0, l: 0, d: 0 }
    l.w += r.wins || 0
    l.l += r.losses || 0
    l.d += r.draws || 0
    map.set(r.learner_id, l)
    // Opponent side (mirror)
    const o = map.get(r.opponent_id) || { w: 0, l: 0, d: 0 }
    o.w += r.losses || 0
    o.l += r.wins || 0
    o.d += r.draws || 0
    map.set(r.opponent_id, o)
  }
  return map
})

export const leagueRanked = derived(leagueEntries, ($entries) => {
  const sorted = [...$entries].sort((a, b) => b.elo_rating - a.elo_rating)
  return sorted.map((entry, i) => ({ ...entry, rank: i + 1 }))
})

/** League-level summary stats */
export const leagueStats = derived(
  [leagueEntries, leagueResults],
  ([$entries, $results]) => {
    if ($entries.length === 0) return null
    const elos = $entries.map(e => e.elo_rating)
    const sorted = [...$entries].sort((a, b) => b.elo_rating - a.elo_rating)
    const totalMatches = $results.length
    return {
      poolSize: $entries.length,
      totalMatches,
      topEntry: sorted[0],
      eloMin: Math.round(Math.min(...elos)),
      eloMax: Math.round(Math.max(...elos)),
      eloSpread: Math.round(Math.max(...elos) - Math.min(...elos)),
    }
  }
)
