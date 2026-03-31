# Plan 7: App Shell + Status Indicator

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble the full dashboard layout — header with training status, game thumbnails on the left, focused board + move log on the right, metrics grid at the bottom.

**Architecture:** `App.svelte` is the layout shell. It imports all components, connects the WebSocket on mount, and wires stores to components. The layout matches the spec's "Layout B" wireframe.

**Tech Stack:** Svelte

---

### Task 1: Training Status Indicator

**Files:**
- Create: `webui/src/lib/StatusIndicator.svelte`

- [ ] **Step 1: Implement StatusIndicator.svelte**

`webui/src/lib/StatusIndicator.svelte`:
```svelte
<script>
  import { trainingState, trainingAlive } from '../stores/training.js'

  $: status = $trainingState?.status || 'unknown'
  $: epoch = $trainingState?.current_epoch || 0
  $: alive = $trainingAlive
  $: displayName = $trainingState?.display_name || 'Player'

  $: indicator = alive
    ? { dot: 'green', text: `Training alive (epoch ${epoch})` }
    : status === 'completed'
      ? { dot: 'red', text: 'Training completed' }
      : status === 'paused'
        ? { dot: 'red', text: 'Training paused' }
        : { dot: 'yellow', text: 'Training stale' }
</script>

<div class="status-bar">
  <div class="left">
    <h1>Keisei Training Dashboard</h1>
    <div class="indicator">
      <span class="dot" style="background: {indicator.dot === 'green' ? 'var(--accent-green)' : indicator.dot === 'yellow' ? 'var(--warning)' : 'var(--danger)'}"></span>
      <span class="text">{indicator.text}</span>
    </div>
  </div>
  <div class="right">
    <span class="player-name">☗ {displayName}</span>
  </div>
</div>

<style>
  .status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
  }

  .left {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  h1 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
  }

  .right {
    font-size: 14px;
  }

  .player-name {
    color: var(--accent-green);
    font-weight: 600;
  }
</style>
```

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/StatusIndicator.svelte
git commit -m "feat: training status indicator with heartbeat-based liveness"
```

---

### Task 2: App.svelte Layout

**Files:**
- Modify: `webui/src/App.svelte`

- [ ] **Step 1: Implement full App.svelte layout**

`webui/src/App.svelte`:
```svelte
<script>
  import { onMount } from 'svelte'
  import { connect } from './lib/ws.js'
  import { games, selectedGame } from './stores/games.js'
  import StatusIndicator from './lib/StatusIndicator.svelte'
  import GameThumbnail from './lib/GameThumbnail.svelte'
  import Board from './lib/Board.svelte'
  import PieceTray from './lib/PieceTray.svelte'
  import MoveLog from './lib/MoveLog.svelte'
  import MetricsGrid from './lib/MetricsGrid.svelte'

  onMount(() => {
    connect()
  })

  // Parse selected game data
  $: game = $selectedGame
  $: board = game
    ? (typeof game.board_json === 'string' ? JSON.parse(game.board_json) : (game.board || []))
    : []
  $: hands = game
    ? (typeof game.hands_json === 'string' ? JSON.parse(game.hands_json) : (game.hands || {}))
    : {}
  $: moveHistory = game?.move_history_json || '[]'
</script>

<div class="app">
  <StatusIndicator />

  <div class="main-content">
    <!-- Left: Game thumbnails -->
    <aside class="thumbnail-panel">
      <div class="thumb-label">Games ({$games.length})</div>
      <div class="thumb-grid">
        {#each $games as g (g.game_id)}
          <GameThumbnail game={g} />
        {/each}
      </div>
    </aside>

    <!-- Center + Right: Focus board + game info -->
    <section class="game-panel">
      {#if game}
        <div class="game-view">
          <div class="board-area">
            <PieceTray color="white" hand={hands.white || {}} />
            <Board
              board={board}
              inCheck={!!game.in_check}
              currentPlayer={game.current_player || 'black'}
            />
            <PieceTray color="black" hand={hands.black || {}} />
          </div>

          <div class="info-area">
            <div class="game-info">
              <div class="info-row">
                <span class="label">Game {(game.game_id || 0) + 1}</span>
                <span class="value">{game.current_player || 'black'} to move</span>
              </div>
              <div class="info-row">
                <span class="label">Ply</span>
                <span class="value">{game.ply || 0}</span>
              </div>
              <div class="info-row">
                <span class="label">Result</span>
                <span class="value result"
                  class:in-progress={game.result === 'in_progress'}
                  class:terminal={game.result !== 'in_progress'}
                >
                  {(game.result || 'in_progress').replace('_', ' ')}
                </span>
              </div>
            </div>

            <MoveLog
              moveHistoryJson={moveHistory}
              currentPlayer={game.current_player || 'black'}
            />
          </div>
        </div>
      {:else}
        <div class="no-game">
          <p>Waiting for game data...</p>
        </div>
      {/if}
    </section>
  </div>

  <!-- Bottom: Metrics -->
  <section class="metrics-panel">
    <MetricsGrid />
  </section>
</div>

<style>
  .app {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    background: var(--bg-primary);
  }

  .main-content {
    display: flex;
    flex: 1;
    gap: 0;
    border-bottom: 1px solid var(--border);
  }

  .thumbnail-panel {
    width: 200px;
    flex-shrink: 0;
    border-right: 1px solid var(--border);
    padding: 8px;
    overflow-y: auto;
    max-height: calc(100vh - 200px);
  }

  .thumb-label {
    font-size: 10px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }

  .thumb-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }

  .game-panel {
    flex: 1;
    padding: 16px;
    overflow-y: auto;
  }

  .game-view {
    display: flex;
    gap: 16px;
  }

  .board-area {
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
  }

  .info-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 220px;
  }

  .game-info {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
  }

  .info-row {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    font-size: 12px;
  }

  .info-row .label {
    color: var(--text-secondary);
  }

  .info-row .value {
    color: var(--text-primary);
  }

  .result.in-progress { color: var(--accent-amber); }
  .result.terminal { color: var(--accent-green); }

  .no-game {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    color: var(--text-muted);
  }

  .metrics-panel {
    padding: 12px 16px;
  }
</style>
```

- [ ] **Step 2: Verify dev server renders the layout**

```bash
cd webui && npm run dev
# Open http://localhost:5173 — should show layout skeleton
# (No live data unless FastAPI server is running with training data)
```

- [ ] **Step 3: Commit**

```bash
git add webui/src/App.svelte webui/src/lib/StatusIndicator.svelte
git commit -m "feat: full dashboard layout with game panel, thumbnails, and metrics"
```
