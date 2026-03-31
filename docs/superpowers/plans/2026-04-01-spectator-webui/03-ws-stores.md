# Plan 3: WebSocket Client + Svelte Stores

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the WebSocket client that connects to the FastAPI server and Svelte stores that hold game state, metrics, and training status reactively.

**Architecture:** A single `ws.js` module manages the WebSocket connection (connect, reconnect, dispatch messages). Svelte writable stores hold the data. On reconnect, stores are replaced (not merged) per spec.

**Tech Stack:** Svelte stores, WebSocket API

---

### Task 1: WebSocket Client

**Files:**
- Create: `webui/src/lib/ws.js`

- [ ] **Step 1: Implement WebSocket client**

`webui/src/lib/ws.js`:
```js
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
      // Replace all state on init (reconnect = full reset)
      games.set(msg.games || [])
      metrics.set(msg.metrics || [])
      trainingState.set(msg.training_state || null)
      // Auto-select first game if none selected
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
      // Server ping — no action needed
      break
  }
}

export function disconnect() {
  if (ws) {
    ws.close()
    ws = null
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/ws.js
git commit -m "feat: WebSocket client with auto-reconnect and message dispatch"
```

---

### Task 2: Svelte Stores

**Files:**
- Create: `webui/src/stores/games.js`
- Create: `webui/src/stores/metrics.js`
- Create: `webui/src/stores/training.js`

- [ ] **Step 1: Create games store**

`webui/src/stores/games.js`:
```js
import { writable, derived } from 'svelte/store'

/** Array of game snapshot objects from the server. */
export const games = writable([])

/** Currently selected game index for the focus view. */
export const selectedGameId = writable(0)

/** Derived: the currently selected game snapshot. */
export const selectedGame = derived(
  [games, selectedGameId],
  ([$games, $id]) => $games.find(g => g.game_id === $id) || $games[0] || null
)
```

- [ ] **Step 2: Create metrics store**

`webui/src/stores/metrics.js`:
```js
import { writable, derived } from 'svelte/store'

const MAX_POINTS = 10000

/** Raw metrics rows from the server. */
export const metrics = writable([])

// Cap the store at MAX_POINTS to prevent unbounded memory growth.
// Older points are dropped (the server sends full history on init,
// but we only keep the tail for rendering).
metrics.subscribe(rows => {
  if (rows.length > MAX_POINTS) {
    metrics.set(rows.slice(-MAX_POINTS))
  }
})

/** Derived: latest metrics row for KPI display. */
export const latestMetrics = derived(metrics, $m => $m[$m.length - 1] || null)
```

- [ ] **Step 3: Create training state store**

`webui/src/stores/training.js`:
```js
import { writable, derived } from 'svelte/store'

/** Singleton training state from the server. */
export const trainingState = writable(null)

/** Derived: is training alive (heartbeat fresh)? */
export const trainingAlive = derived(trainingState, $s => {
  if (!$s || !$s.heartbeat_at) return false
  const hbTime = new Date($s.heartbeat_at).getTime()
  const age = Date.now() - hbTime
  return age < 30000 // 30 seconds
})
```

- [ ] **Step 4: Commit**

```bash
git add webui/src/stores/
git commit -m "feat: Svelte stores for games, metrics, and training state"
```
