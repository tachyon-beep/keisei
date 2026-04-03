import { writable, derived } from 'svelte/store'
import { leagueEntries } from './league.js'

export const games = writable([])
export const selectedGameId = writable(0)
export const selectedGame = derived(
  [games, selectedGameId],
  ([$games, $id]) => $games.find(g => g.game_id === $id) || $games[0] || null
)

export const selectedOpponent = derived(
  [selectedGame, leagueEntries],
  ([$game, $entries]) => {
    if (!$game?.opponent_id) return null
    const entry = $entries.find(e => e.id === $game.opponent_id)
    if (!entry) return null
    return {
      display_name: entry.display_name || entry.architecture,
      architecture: entry.architecture,
      elo_rating: entry.elo_rating,
      games_played: entry.games_played,
      created_epoch: entry.created_epoch,
      flavour_facts: entry.flavour_facts || [],
      created_at: entry.created_at || '',
    }
  }
)
