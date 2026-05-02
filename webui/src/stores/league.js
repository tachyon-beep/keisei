import { writable, derived } from 'svelte/store'
import { get } from 'svelte/store'
import { trainingState } from './training.js'

export const leagueEntries = writable([])
export const leagueResults = writable([])
export const eloHistory = writable([])
export const tournamentStats = writable(null)
export const styleProfilesRaw = writable([])
/** Raw head-to-head aggregates from backend (canonical ordering: entry_a_id < entry_b_id) */
export const headToHeadRaw = writable([])

/** Currently expanded/focused entry in leaderboard — used for cross-highlighting */
export const focusedEntryId = writable(null)

export const historicalLibrary = writable([])
export const gauntletResults = writable([])
export const leagueTransitions = writable([])

/** Event log: tracks arrivals, departures, and rank changes.
 *  Persisted to localStorage so events survive page refresh.
 *
 *  Run discrimination: events from a previous training run are stale and
 *  must be cleared.  We persist a "run marker" (set of (id, display_name)
 *  fingerprints from the entry pool) alongside events.  On first load:
 *    - Any fingerprint overlap with current entries -> same run, keep events.
 *    - No overlap (or no marker) -> new run (fresh DB, league reset, etc.) ->
 *      clear stale events.
 *  Display names are randomly generated per entry, so cross-run collisions
 *  are effectively impossible even if IDs replay after a DB wipe. */
const MAX_EVENTS = 50
const EVENTS_STORAGE_KEY = 'keisei_league_events'
const RUN_MARKER_STORAGE_KEY = 'keisei_league_event_run_marker'

function loadPersistedEvents() {
  try {
    const raw = localStorage.getItem(EVENTS_STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch { return [] }
}

function loadPersistedRunMarker() {
  try {
    const raw = localStorage.getItem(RUN_MARKER_STORAGE_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed : []
  } catch { return [] }
}

function persistRunMarker(fingerprints) {
  try {
    localStorage.setItem(RUN_MARKER_STORAGE_KEY, JSON.stringify(fingerprints))
  } catch {}
}

function entryFingerprints(entries) {
  return entries
    .filter(e => e.display_name)
    .map(e => `${e.id}:${e.display_name}`)
}

export const leagueEvents = writable(loadPersistedEvents())

leagueEvents.subscribe(events => {
  try { localStorage.setItem(EVENTS_STORAGE_KEY, JSON.stringify(events)) } catch {}
})

let _prevEntryMap = new Map()
let _prevRanks = new Map()
let _persistedFingerprints = loadPersistedRunMarker()

/** Clear persisted events and diff state. Used in tests to reset
 *  module-level mutable state after vi.resetModules(). */
export function resetLeagueEvents() {
  _prevEntryMap = new Map()
  _prevRanks = new Map()
  _persistedFingerprints = []
  persistRunMarker([])
  leagueEvents.set([])
}

/**
 * Call after leagueEntries is updated to diff and generate events.
 * Kept as an explicit function (not a derived store) because it needs
 * to accumulate history across updates rather than recompute from scratch.
 */
export function diffLeagueEntries(entries) {
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  const newMap = new Map(entries.map(e => [e.id, e]))

  // First load: seed state without generating events (avoids phantom
  // arrivals for every existing entry on page refresh).
  if (_prevEntryMap.size === 0) {
    // Discriminate same-run refresh from new-run startup using the
    // (id, display_name) fingerprint marker.  Same run -> keep persisted
    // events.  New run (DB wipe, league reset, switched config) -> drop
    // stale events from the previous run.
    const newFingerprints = entryFingerprints(entries)
    const newFpSet = new Set(newFingerprints)
    const sameRun = _persistedFingerprints.length > 0 &&
                    _persistedFingerprints.some(fp => newFpSet.has(fp))
    if (!sameRun) {
      leagueEvents.set([])
    }
    _prevEntryMap = newMap
    _persistedFingerprints = newFingerprints
    persistRunMarker(newFingerprints)

    const activeEntries = entries.filter(e => e.status === 'active')
    if (activeEntries.length > 0) {
      const sorted = [...activeEntries].sort((a, b) => b.elo_rating - a.elo_rating)
      _prevRanks = new Map(sorted.map((e, i) => [e.id, i + 1]))
    }
    return
  }

  const events = []

  // Arrivals (active entries only)
  for (const [id, entry] of newMap) {
    if (!_prevEntryMap.has(id) && entry.status === 'active') {
      events.push({
        time: now,
        type: 'arrival',
        icon: '→',
        name: entry.display_name || entry.architecture,
        role: entry.role,
        detail: `joined at Elo ${Math.round(entry.elo_rating)}`,
      })
    }
  }

  // Retirements (status changed from active to retired)
  for (const [id, entry] of newMap) {
    const prev = _prevEntryMap.get(id)
    if (prev && prev.status === 'active' && entry.status === 'retired') {
      events.push({
        time: now,
        type: 'departure',
        icon: '←',
        name: entry.display_name || entry.architecture,
        role: entry.role,
        detail: 'retired',
      })
    }
  }

  // Departures (deleted from DB entirely — should be rare)
  for (const [id, entry] of _prevEntryMap) {
    if (!newMap.has(id)) {
      events.push({
        time: now,
        type: 'departure',
        icon: '←',
        name: entry.display_name || entry.architecture,
        role: entry.role,
        detail: 'removed from pool',
      })
    }
  }

  // Rank changes (only active entries, not first load)
  const activeEntries = entries.filter(e => e.status === 'active')
  if (_prevRanks.size > 0 && activeEntries.length > 1) {
    const sorted = [...activeEntries].sort((a, b) => b.elo_rating - a.elo_rating)
    for (let i = 0; i < sorted.length; i++) {
      const e = sorted[i]
      const newRank = i + 1
      const oldRank = _prevRanks.get(e.id)
      if (oldRank != null && oldRank !== newRank) {
        // Only log significant rank changes (top 3 movements)
        if (newRank <= 3 || oldRank <= 3) {
          const direction = newRank < oldRank ? '↑' : '↓'
          events.push({
            time: now,
            type: newRank < oldRank ? 'promotion' : 'demotion',
            icon: direction,
            name: e.display_name || e.architecture,
            role: e.role,
            detail: `rank ${oldRank} → ${newRank}`,
          })
        }
      }
    }
    // Update rank cache
    _prevRanks = new Map(sorted.map((e, i) => [e.id, i + 1]))
  }

  _prevEntryMap = newMap
  // Refresh the run marker so a refresh mid-run can detect the current
  // entry set, not whatever was around the last time an event fired.
  _persistedFingerprints = entryFingerprints(entries)
  persistRunMarker(_persistedFingerprints)

  if (events.length > 0) {
    leagueEvents.update(existing => [...events, ...existing].slice(0, MAX_EVENTS))
  }
}

/** Aggregate W/L/D totals keyed by entry id */
export const entryWLD = derived(leagueResults, ($results) => {
  const map = new Map()
  for (const r of $results) {
    // Side A
    const a = map.get(r.entry_a_id) || { w: 0, l: 0, d: 0 }
    a.w += r.wins_a || 0
    a.l += r.wins_b || 0
    a.d += r.draws || 0
    map.set(r.entry_a_id, a)
    // Side B (mirror)
    const b = map.get(r.entry_b_id) || { w: 0, l: 0, d: 0 }
    b.w += r.wins_b || 0
    b.l += r.wins_a || 0
    b.d += r.draws || 0
    map.set(r.entry_b_id, b)
  }
  return map
})

export const leagueRanked = derived(leagueEntries, ($entries) => {
  const active = $entries.filter(e => e.status === 'active')
  const sorted = [...active].sort((a, b) => b.elo_rating - a.elo_rating)
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

/** Head-to-head win rates: Map<"attackerId-defenderId", { w, l, d, winRate, total }>
 *  Built from pre-aggregated backend data (headToHeadRaw) instead of raw results.
 *  Backend sends canonical ordering (a < b); we expand to bidirectional keys here.
 */
export const headToHead = derived(headToHeadRaw, ($h2hRaw) => {
  const map = new Map()
  for (const r of $h2hRaw) {
    // A vs B (canonical direction from backend)
    const keyAB = `${r.entry_a_id}-${r.entry_b_id}`
    const totalGames = r.games || (r.wins_a + r.wins_b + r.draws)
    const winRateAB = totalGames > 0 ? r.wins_a / totalGames : null
    map.set(keyAB, {
      w: r.wins_a,
      l: r.wins_b,
      d: r.draws,
      total: totalGames,
      winRate: winRateAB,
    })
    // B vs A (mirror)
    const keyBA = `${r.entry_b_id}-${r.entry_a_id}`
    const winRateBA = totalGames > 0 ? r.wins_b / totalGames : null
    map.set(keyBA, {
      w: r.wins_b,
      l: r.wins_a,
      d: r.draws,
      total: totalGames,
      winRate: winRateBA,
    })
  }
  return map
})

/** League-level summary stats */
export const leagueStats = derived(
  [leagueEntries, leagueResults],
  ([$entries, $results]) => {
    const active = $entries.filter(e => e.status === 'active')
    if (active.length === 0) return null
    const elos = active.map(e => displayElo(e).value)
    const sorted = [...active].sort((a, b) => displayElo(b).value - displayElo(a).value)
    const totalMatches = $results.length
    const totalRounds = new Set($results.map(r => r.epoch)).size
    const totalGames = $results.reduce((sum, r) => sum + (r.wins_a || 0) + (r.wins_b || 0) + (r.draws || 0), 0)
    return {
      poolSize: active.length,
      totalMatches,
      totalRounds,
      totalGames,
      topEntry: sorted[0],
      eloMin: Math.round(Math.min(...elos)),
      eloMax: Math.round(Math.max(...elos)),
      eloSpread: Math.round(Math.max(...elos) - Math.min(...elos)),
    }
  }
)

/** The league entry matching the current learner (by learner_entry_id from trainingState, with display_name fallback) */
export const learnerEntry = derived(
  [leagueEntries, trainingState],
  ([$entries, $state]) => {
    if (!$state) return null
    const id = $state.learner_entry_id
    if (id != null) {
      return $entries.find(e => e.id === id) || null
    }
    // Fallback for older backends that don't send learner_entry_id
    const name = $state.display_name
    if (!name) return null
    return $entries.find(e => e.display_name === name) || null
  }
)

/** Style profiles keyed by checkpoint_id for quick lookup */
export const styleProfiles = derived(styleProfilesRaw, ($profiles) => {
  const map = new Map()
  for (const p of $profiles) {
    map.set(p.checkpoint_id, p)
  }
  return map
})

const ROLE_ELO_COLUMN = {
  frontier_static: 'elo_frontier',
  dynamic: 'elo_dynamic',
  recent_fixed: 'elo_recent',
}
const ROLE_ELO_TAG = {
  frontier_static: 'F',
  dynamic: 'D',
  recent_fixed: 'R',
}

/** Return { value, tag } for the entry's displayed Elo (role-specific when available). */
export function displayElo(entry) {
  const col = ROLE_ELO_COLUMN[entry.role]
  const roleVal = col ? entry[col] : null
  if (roleVal != null && roleVal !== 1000) {
    return { value: roleVal, tag: ROLE_ELO_TAG[entry.role] || '' }
  }
  return { value: entry.elo_rating, tag: '' }
}

const KNOWN_ROLES = new Set(['frontier_static', 'recent_fixed', 'dynamic', 'historical'])

/** Groups active entries by role. Unknown/null roles go to 'other'. */
export const leagueByRole = derived(leagueRanked, ($ranked) => {
  const map = new Map()
  for (const entry of $ranked) {
    const key = KNOWN_ROLES.has(entry.role) ? entry.role : 'other'
    if (!map.has(key)) map.set(key, [])
    map.get(key).push(entry)
  }
  return map
})

const PROMOTION_PAIRS = new Set([
  'recent_fixed->frontier_static',
  'dynamic->frontier_static',
  'dynamic->recent_fixed',
])

/** Counts promotions, evictions, and admissions from backend transition records. */
export const transitionCounts = derived(leagueTransitions, ($transitions) => {
  let promotions = 0, evictions = 0, admissions = 0
  for (const t of $transitions) {
    // Status transitions take precedence
    if (!t.from_status && t.to_status === 'active') {
      admissions++
    } else if (t.from_status === 'active' && t.to_status === 'retired') {
      evictions++
    } else if (
      t.from_status === 'active' && t.to_status === 'active' &&
      PROMOTION_PAIRS.has(`${t.from_role}->${t.to_role}`)
    ) {
      promotions++
    }
  }
  return { promotions, evictions, admissions }
})
