import { writable, derived } from 'svelte/store'

const MAX_POINTS = 10000

function prune(arr) {
  return arr.length > MAX_POINTS ? arr.slice(-MAX_POINTS) : arr
}

function createMetricsStore() {
  const { subscribe, set, update } = writable([])
  return {
    subscribe,
    set: (value) => set(prune(value)),
    update: (fn) => update(current => prune(fn(current))),
  }
}

export const metrics = createMetricsStore()

/** Append new metrics rows and trim to MAX_POINTS. Called from ws.js. */
export function appendMetrics(newRows) {
  metrics.update(current => [...current, ...newRows])
}

export const latestMetrics = derived(metrics, $m => $m[$m.length - 1] || null)
