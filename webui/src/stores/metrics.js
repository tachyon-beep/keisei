import { writable, derived } from 'svelte/store'

const MAX_POINTS = 10000

export const metrics = writable([])

metrics.subscribe(rows => {
  if (rows.length > MAX_POINTS) {
    metrics.set(rows.slice(-MAX_POINTS))
  }
})

export const latestMetrics = derived(metrics, $m => $m[$m.length - 1] || null)
