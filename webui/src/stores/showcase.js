import { writable, derived, get } from 'svelte/store'

/** Active showcase game metadata */
export const showcaseGame = writable(null)

/** All moves in the current showcase game */
export const showcaseMoves = writable([])

/** Queue of pending/running showcase matches */
export const showcaseQueue = writable([])

/** Whether the sidecar process is alive */
export const sidecarAlive = writable(false)

/**
 * Selected ply index (0-based), or null to follow the latest move ("live").
 * When set, the spectator is scrubbing through history. Reset to null when
 * the active game changes — see resetShowcaseSelectionOnGameChange below.
 */
export const showcaseSelectedPly = writable(null)

/** Most recent move (last entry in showcaseMoves), or null if no moves yet. */
export const showcaseCurrentMove = derived(showcaseMoves, moves => {
  if (moves.length === 0) return null
  return moves[moves.length - 1]
})

/**
 * The move actually being displayed: the selected ply if scrubbing, else the
 * latest move. Use this everywhere the board/eval/commentary should reflect
 * what the user is *looking at*, not what's necessarily live.
 */
export const showcaseDisplayedMove = derived(
  [showcaseMoves, showcaseSelectedPly],
  ([moves, selectedIdx]) => {
    if (moves.length === 0) return null
    if (selectedIdx == null) return moves[moves.length - 1]
    const clamped = Math.max(0, Math.min(selectedIdx, moves.length - 1))
    return moves[clamped]
  }
)

/** True when the user has scrubbed off "live" — i.e. selectedPly is set. */
export const isScrubbing = derived(
  [showcaseMoves, showcaseSelectedPly],
  ([moves, selectedIdx]) => selectedIdx != null && selectedIdx < moves.length - 1
)

/** Win probability history for the graph */
export const winProbHistory = derived(showcaseMoves, moves => {
  return moves.map(m => ({ ply: m.ply, value: m.value_estimate }))
})

/** Number of pending matches in queue */
export const queueDepth = derived(showcaseQueue, q =>
  q.filter(e => e.status === 'pending').length
)

/** Persisted toggle for the showcase board's policy heatmap overlay. */
const HEATMAP_KEY = 'showcaseHeatmapEnabled'

function loadHeatmapInitial() {
  if (typeof localStorage === 'undefined') return false
  return localStorage.getItem(HEATMAP_KEY) === 'true'
}

export const showcaseHeatmapEnabled = writable(loadHeatmapInitial())

showcaseHeatmapEnabled.subscribe((val) => {
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(HEATMAP_KEY, val ? 'true' : 'false')
  }
})

/** Persisted speed setting for the showcase engine. */
const SPEED_KEY = 'showcaseSpeed'
const VALID_SPEEDS = new Set(['slow', 'normal', 'fast'])

function loadSpeedInitial() {
  if (typeof localStorage === 'undefined') return 'normal'
  const v = localStorage.getItem(SPEED_KEY)
  return VALID_SPEEDS.has(v) ? v : 'normal'
}

export const showcaseSpeed = writable(loadSpeedInitial())

showcaseSpeed.subscribe((val) => {
  if (typeof localStorage !== 'undefined' && VALID_SPEEDS.has(val)) {
    localStorage.setItem(SPEED_KEY, val)
  }
})

/**
 * Reset the scrub selection back to "live" whenever a new game starts. Call
 * this from the WS handler after assigning showcaseGame; otherwise a stale
 * index pinned during the previous match would point into the wrong move
 * history (or would index out of bounds during the gap).
 */
export function resetShowcaseSelectionOnGameChange(newGameId) {
  // Always clear on new/null game; the previous selection has no meaning.
  showcaseSelectedPly.set(null)
}
