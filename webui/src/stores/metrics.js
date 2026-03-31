import { writable, derived } from 'svelte/store'

const MAX_POINTS = 10000

export const metrics = writable([])

/** Append new metrics rows and trim to MAX_POINTS. Called from ws.js. */
export function appendMetrics(newRows) {
  metrics.update(current => {
    const combined = [...current, ...newRows]
    return combined.length > MAX_POINTS ? combined.slice(-MAX_POINTS) : combined
  })
}

export const latestMetrics = derived(metrics, $m => $m[$m.length - 1] || null)
