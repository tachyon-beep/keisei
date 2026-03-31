/**
 * WebSocket client for Keisei spectator dashboard.
 * Connects to /ws, dispatches messages to Svelte stores.
 * Auto-reconnects on disconnect with exponential backoff.
 */

import { games, selectedGameId } from '../stores/games.js'
import { metrics } from '../stores/metrics.js'
import { trainingState } from '../stores/training.js'

const WS_URL = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`
const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 30000

let ws = null
let reconnectAttempt = 0

export function connect() {
  if (ws && ws.readyState <= WebSocket.OPEN) return

  ws = new WebSocket(WS_URL)

  ws.onopen = () => {
    console.log('[ws] connected')
    reconnectAttempt = 0
  }

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data)
    handleMessage(msg)
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
  const delay = Math.min(
    RECONNECT_BASE_MS * Math.pow(2, reconnectAttempt),
    RECONNECT_MAX_MS
  )
  reconnectAttempt++
  setTimeout(connect, delay)
}

function handleMessage(msg) {
  switch (msg.type) {
    case 'init':
      games.set(msg.games || [])
      metrics.set(msg.metrics || [])
      trainingState.set(msg.training_state || null)
      if (msg.games?.length > 0) {
        selectedGameId.update(id => id ?? 0)
      }
      break

    case 'game_update':
      games.set(msg.snapshots || [])
      break

    case 'metrics_update':
      metrics.update(current => [...current, ...(msg.rows || [])])
      break

    case 'training_status':
      trainingState.update(state => ({
        ...state,
        status: msg.status,
        heartbeat_at: msg.heartbeat_at,
        current_epoch: msg.epoch,
      }))
      break

    case 'ping':
      break
  }
}

export function disconnect() {
  if (ws) {
    ws.close()
    ws = null
  }
}
