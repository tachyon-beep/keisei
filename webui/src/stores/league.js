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

/** Elo delta: difference between latest and earliest recorded Elo per entry */
export const eloDelta = derived(eloHistory, ($history) => {
  const map = new Map()
  for (const h of $history) {
    const rec = map.get(h.entry_id)
    if (!rec) {
      map.set(h.entry_id, { first: h.elo_rating, firstEpoch: h.epoch, last: h.elo_rating, lastEpoch: h.epoch })
    } else {
      if (h.epoch < rec.firstEpoch) { rec.first = h.elo_rating; rec.firstEpoch = h.epoch }
      if (h.epoch > rec.lastEpoch) { rec.last = h.elo_rating; rec.lastEpoch = h.epoch }
    }
  }
  const deltas = new Map()
  for (const [id, rec] of map) {
    deltas.set(id, Math.round(rec.last - rec.first))
  }
  return deltas
})

/** Head-to-head win rates: Map<"attackerId-defenderId", { w, l, d, winRate }> */
export const headToHead = derived(leagueResults, ($results) => {
  const map = new Map()
  for (const r of $results) {
    // Learner vs Opponent
    const keyLO = `${r.learner_id}-${r.opponent_id}`
    const lo = map.get(keyLO) || { w: 0, l: 0, d: 0 }
    lo.w += r.wins || 0
    lo.l += r.losses || 0
    lo.d += r.draws || 0
    map.set(keyLO, lo)
    // Opponent vs Learner (mirror)
    const keyOL = `${r.opponent_id}-${r.learner_id}`
    const ol = map.get(keyOL) || { w: 0, l: 0, d: 0 }
    ol.w += r.losses || 0
    ol.l += r.wins || 0
    ol.d += r.draws || 0
    map.set(keyOL, ol)
  }
  // Compute win rates
  for (const [key, rec] of map) {
    const total = rec.w + rec.l + rec.d
    rec.winRate = total > 0 ? rec.w / total : null
    rec.total = total
  }
  return map
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
