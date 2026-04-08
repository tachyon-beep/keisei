import { writable, derived } from 'svelte/store'

/** Active showcase game metadata */
export const showcaseGame = writable(null)

/** All moves in the current showcase game */
export const showcaseMoves = writable([])

/** Queue of pending/running showcase matches */
export const showcaseQueue = writable([])

/** Current speed setting */
export const showcaseSpeed = writable('normal')

/** Whether the sidecar process is alive */
export const sidecarAlive = writable(false)

/** Full move object for the most recent move (board, candidates, eval, etc.) */
export const showcaseCurrentMove = derived(showcaseMoves, moves => {
  if (moves.length === 0) return null
  return moves[moves.length - 1]
})

/** Win probability history for the graph */
export const winProbHistory = derived(showcaseMoves, moves => {
  return moves.map(m => ({ ply: m.ply, value: m.value_estimate }))
})

/** Number of pending matches in queue */
export const queueDepth = derived(showcaseQueue, q =>
  q.filter(e => e.status === 'pending').length
)
