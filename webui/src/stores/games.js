import { writable, derived } from 'svelte/store'

export const games = writable([])
export const selectedGameId = writable(0)
export const selectedGame = derived(
  [games, selectedGameId],
  ([$games, $id]) => $games.find(g => g.game_id === $id) || $games[0] || null
)
