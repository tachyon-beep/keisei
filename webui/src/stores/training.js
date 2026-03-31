import { writable, derived } from 'svelte/store'

export const trainingState = writable(null)

export const trainingAlive = derived(trainingState, $s => {
  if (!$s || !$s.heartbeat_at) return false
  const hbTime = new Date($s.heartbeat_at).getTime()
  const age = Date.now() - hbTime
  return age < 30000
})
