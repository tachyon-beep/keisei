import { writable, derived } from 'svelte/store'

export const trainingState = writable(null)

// Tick every 10s to re-evaluate heartbeat freshness
const tick = writable(Date.now())
if (typeof window !== 'undefined') {
  setInterval(() => tick.set(Date.now()), 10000)
}

export const trainingAlive = derived(
  [trainingState, tick],
  ([$s, $now]) => {
    if (!$s || !$s.heartbeat_at) return false
    const hbTime = new Date($s.heartbeat_at).getTime()
    const age = $now - hbTime
    return age < 30000
  }
)
