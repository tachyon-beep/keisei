<script>
  import { onMount, onDestroy } from 'svelte'
  import { connect, disconnect } from './lib/ws.js'
  import { games, selectedGame } from './stores/games.js'
  import StatusIndicator from './lib/StatusIndicator.svelte'
  import GameThumbnail from './lib/GameThumbnail.svelte'
  import Board from './lib/Board.svelte'
  import PieceTray from './lib/PieceTray.svelte'
  import MoveLog from './lib/MoveLog.svelte'
  import MetricsGrid from './lib/MetricsGrid.svelte'

  onMount(() => {
    connect()
    return disconnect
  })

  function safeParse(json, fallback) {
    try { return typeof json === 'string' ? JSON.parse(json) : json }
    catch { return fallback }
  }

  $: game = $selectedGame
  $: board = game ? safeParse(game.board_json, game.board || []) : []
  $: hands = game ? safeParse(game.hands_json, game.hands || {}) : {}
  $: moveHistory = game?.move_history_json || '[]'

  $: lastMoveIdx = (() => {
    try {
      const history = safeParse(moveHistory, [])
      if (history.length === 0) return -1
      const lastAction = history[history.length - 1]
      // We don't have direct square mapping from action index,
      // so use -1 for now (can be enhanced when action mapper is available)
      return -1
    } catch { return -1 }
  })()
</script>

<div class="app">
  <a href="#game-panel" class="skip-nav">Skip to game</a>
  <StatusIndicator />

  <div class="main-content">
    <aside class="thumbnail-panel" aria-label="Game list">
      <h2 class="thumb-label">Games ({$games.length})</h2>
      <div class="thumb-grid">
        {#each $games as g (g.game_id)}
          <GameThumbnail game={g} />
        {/each}
      </div>
    </aside>

    <main id="game-panel" class="game-panel" aria-label="Game viewer">
      {#if game}
        <div class="game-view">
          <div class="board-area">
            <PieceTray color="white" hand={hands.white || {}} />
            <Board
              board={board}
              inCheck={!!game.in_check}
              currentPlayer={game.current_player || 'black'}
              lastMoveIdx={lastMoveIdx}
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
                  {#if game.result === 'in_progress'}In progress{:else}&#10003; {(game.result || '').replaceAll('_', ' ')}{/if}
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
          <p>Waiting for game data&hellip;</p>
          <p class="no-game-hint">Connect a training session to see live games.</p>
        </div>
      {/if}
    </main>
  </div>

  <section class="metrics-panel" aria-label="Training metrics">
    <MetricsGrid />
  </section>
</div>

<style>
  .skip-nav {
    position: absolute;
    left: -9999px;
    top: 0;
    z-index: 100;
    padding: 8px 16px;
    background: var(--accent-blue);
    color: #fff;
    font-size: 14px;
    font-weight: 600;
    text-decoration: none;
    border-radius: 0 0 4px 0;
  }

  .skip-nav:focus {
    left: 0;
  }

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
  }

  h2.thumb-label {
    font-size: 12px;
    font-weight: 600;
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
    background: var(--bg-primary);
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
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    color: var(--text-muted);
    gap: 8px;
  }

  .no-game-hint {
    font-size: 12px;
    color: var(--text-muted);
  }

  .metrics-panel {
    padding: 12px 16px;
  }

  @media (max-width: 768px) {
    .main-content {
      flex-direction: column;
    }

    .thumbnail-panel {
      width: 100%;
      border-right: none;
      border-bottom: 1px solid var(--border);
      max-height: 160px;
    }

    .thumb-grid {
      grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    }

    .game-view {
      flex-direction: column;
    }

    .board-area {
      align-self: center;
    }

    .info-area {
      min-width: unset;
    }
  }

  @media (max-width: 480px) {
    .game-panel {
      padding: 8px;
    }

    .metrics-panel {
      padding: 8px;
    }
  }
</style>
