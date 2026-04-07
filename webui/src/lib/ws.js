/**
 * WebSocket client for Keisei spectator dashboard.
 * Connects to /ws, dispatches messages to Svelte stores.
 * Auto-reconnects on disconnect with exponential backoff.
 */

import { games, selectedGameId } from '../stores/games.js'
import { metrics, appendMetrics } from '../stores/metrics.js'
import { trainingState } from '../stores/training.js'
import {
  leagueEntries, leagueResults, eloHistory, tournamentStats, diffLeagueEntries,
  historicalLibrary, gauntletResults, leagueTransitions, styleProfilesRaw,
} from '../stores/league.js'

const WS_URL = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`
const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 30000

let ws = null
let reconnectAttempt = 0
let reconnectTimer = null

export function connect() {
  if (ws && ws.readyState <= WebSocket.OPEN) return

  ws = new WebSocket(WS_URL)

  ws.onopen = () => {
    console.log('[ws] connected')
    reconnectAttempt = 0
  }

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data)
      handleMessage(msg)
    } catch (e) {
      console.warn('[ws] failed to parse message:', e)
    }
  }

  ws.onclose = () => {
    console.log('[ws] disconnected, reconnecting...')
    scheduleReconnect()
  }

  ws.onerror = (err) => {
    console.warn('[ws] error', err)
    ws.close()
  }
}

function scheduleReconnect() {
  const base = Math.min(
    RECONNECT_BASE_MS * Math.pow(2, reconnectAttempt),
    RECONNECT_MAX_MS
  )
  // Add jitter: 50-150% of base delay
  const delay = base * (0.5 + Math.random())
  reconnectAttempt++
  reconnectTimer = setTimeout(connect, delay)
}

export function handleMessage(msg) {
  switch (msg.type) {
    case 'init':
      games.set(msg.games || [])
      metrics.set(msg.metrics || [])
      trainingState.set(msg.training_state || null)
      leagueEntries.set(msg.league_entries || [])
      diffLeagueEntries(msg.league_entries || [])
      leagueResults.set(msg.league_results || [])
      eloHistory.set(msg.elo_history || [])
      historicalLibrary.set(msg.historical_library || [])
      gauntletResults.set(msg.gauntlet_results || [])
      leagueTransitions.set(msg.transitions || [])
      if (msg.tournament_stats) tournamentStats.set(msg.tournament_stats)
      if (msg.style_profiles) styleProfilesRaw.set(msg.style_profiles)
      if (msg.games?.length > 0) {
        selectedGameId.update(id => id ?? 0)
      }
      break

    case 'game_update': {
      const snapshots = msg.snapshots || []
      // Merge delta snapshots into existing store (backend sends only changed games)
      games.update(existing => {
        const updated = [...existing]
        for (const snap of snapshots) {
          const idx = updated.findIndex(g => g.game_id === snap.game_id)
          if (idx >= 0) updated[idx] = snap
          else updated.push(snap)
        }
        return updated
      })
      // Auto-switch away from ended games
      selectedGameId.update(id => {
        const current = snapshots.find(g => g.game_id === id)
        if (current && current.is_over) {
          const active = snapshots.find(g => !g.is_over)
          if (active) return active.game_id
        }
        return id
      })
      break
    }

    case 'metrics_update':
      appendMetrics(msg.rows || [])
      break

    case 'training_status':
      trainingState.update(state => ({
        ...state,
        status: msg.status,
        phase: msg.phase || state?.phase,
        heartbeat_at: msg.heartbeat_at,
        current_epoch: msg.epoch,
        current_step: msg.step,
        episodes: msg.episodes ?? state?.episodes,
        config_json: msg.config_json || state?.config_json,
        display_name: msg.display_name || state?.display_name,
        model_arch: msg.model_arch || state?.model_arch,
        total_epochs: msg.total_epochs ?? state?.total_epochs,
        system_stats: msg.system_stats || state?.system_stats,
      }))
      break

    case 'league_update':
      leagueEntries.set(msg.entries || [])
      diffLeagueEntries(msg.entries || [])
      leagueResults.set(msg.results || [])
      eloHistory.set(msg.elo_history || [])
      historicalLibrary.set(msg.historical_library || [])
      gauntletResults.set(msg.gauntlet_results || [])
      leagueTransitions.set(msg.transitions || [])
      if (msg.tournament_stats) tournamentStats.set(msg.tournament_stats)
      if (msg.style_profiles) styleProfilesRaw.set(msg.style_profiles)
      break

    case 'ping':
      break
  }
}

export function disconnect() {
  if (reconnectTimer != null) {
    clearTimeout(reconnectTimer)
    reconnectTimer = null
  }
  if (ws) {
    ws.close()
    ws = null
  }
}
