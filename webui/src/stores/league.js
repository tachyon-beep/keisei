import { writable, derived } from 'svelte/store'

export const leagueEntries = writable([])
export const leagueResults = writable([])
export const eloHistory = writable([])

export const leagueRanked = derived(leagueEntries, ($entries) => {
  const sorted = [...$entries].sort((a, b) => b.elo_rating - a.elo_rating)
  return sorted.map((entry, i) => ({ ...entry, rank: i + 1 }))
})
