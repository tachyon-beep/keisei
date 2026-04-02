<script>
  import { onMount } from 'svelte'
  import { connect, disconnect } from './lib/ws.js'
  import { games, selectedGame, selectedOpponent } from './stores/games.js'
  import { activeTab } from './stores/navigation.js'
  import { trainingState } from './stores/training.js'
  import { leagueEntries } from './stores/league.js'
  import StatusIndicator from './lib/StatusIndicator.svelte'
  import GameThumbnail from './lib/GameThumbnail.svelte'
  import Board from './lib/Board.svelte'
  import PieceTray from './lib/PieceTray.svelte'
  import MoveLog from './lib/MoveLog.svelte'
  import EvalBar from './lib/EvalBar.svelte'
  import MetricsGrid from './lib/MetricsGrid.svelte'
  import PlayerCard from './lib/PlayerCard.svelte'
  import LeagueView from './lib/LeagueView.svelte'
  import ShogiLegend from './lib/ShogiLegend.svelte'
  import { safeParse } from './lib/safeParse.js'

  onMount(() => {
    connect()
    return disconnect
  })

  $: game = $selectedGame
  $: board = game ? safeParse(game.board_json, game.board || []) : []
  $: hands = game ? safeParse(game.hands_json, game.hands || {}) : {}
  $: moveHistory = game?.move_history_json || '[]'

  let boardAreaHeight = 0
  let thumbPanelHeight = 0

  $: lastMoveIdx = (() => {
    try {
      const history = safeParse(moveHistory, [])
      if (history.length === 0) return -1
      return -1
    } catch { return -1 }
  })()

  // Learner info from training state
  $: learnerName = $trainingState?.display_name || $trainingState?.model_arch || 'Learner'
  $: learnerElo = (() => {
    if (!$leagueEntries.length) return null
    const epoch = $trainingState?.current_epoch ?? -1
    const match = [...$leagueEntries]
      .filter(e => e.created_epoch <= epoch)
      .sort((a, b) => b.created_epoch - a.created_epoch)[0]
    return match?.elo_rating ?? null
  })()
  $: learnerDetail = $trainingState
    ? `${$trainingState.model_arch || ''} · Epoch ${$trainingState.current_epoch || 0} · ${($trainingState.current_step || 0).toLocaleString()} steps`
    : ''

  // Opponent info from selected game
  $: opp = $selectedOpponent
  $: opponentName = opp ? opp.architecture : 'Self-play'
  $: opponentElo = opp?.elo_rating ?? null
  $: opponentDetail = opp ? `${opp.games_played} games played` : ''
</script>

<div class="app">
  <a href="#game-panel" class="skip-nav">Skip to game</a>
  <StatusIndicator />

  {#if $activeTab === 'training'}
    <div class="main-content">
      <aside class="thumbnail-panel" aria-label="Game list" bind:clientHeight={thumbPanelHeight} style="width: {thumbPanelHeight - 94}px">
        <h2 class="section-label">Games ({$games.length})</h2>
        <div class="thumb-grid">
          {#each $games.slice(0, 16) as g (g.game_id)}
            <GameThumbnail game={g} />
          {/each}
        </div>
      </aside>

      <div class="player-panel">
        <PlayerCard role="learner" name={learnerName} elo={learnerElo} detail={learnerDetail} />
        <div class="vs-separator">VS</div>
        <PlayerCard role="opponent" name={opponentName} elo={opponentElo} detail={opponentDetail} />
      </div>

      <main id="game-panel" class="game-panel" aria-label="Game viewer">
        {#if game}
          <div class="game-view">
            <div class="board-area" bind:clientHeight={boardAreaHeight}>
              <PieceTray color="white" hand={hands.white || {}} />
              <Board
                board={board}
                inCheck={!!game.in_check}
                currentPlayer={game.current_player || 'black'}
                lastMoveIdx={lastMoveIdx}
              />
              <PieceTray color="black" hand={hands.black || {}} />
            </div>

            <div class="eval-area" style="height: {boardAreaHeight}px">
              <EvalBar
                value={game.value_estimate || 0}
                currentPlayer={game.current_player || 'black'}
              />
            </div>

            <div class="info-area" style="height: {boardAreaHeight}px">
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

            <div class="legend-area" style="height: {boardAreaHeight}px">
              <ShogiLegend />
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
  {:else}
    <LeagueView />
  {/if}
</div>

<style>
  .skip-nav {
    position: absolute;
    left: -9999px;
    top: 0;
    z-index: 100;
    padding: 8px 16px;
    background: var(--accent-ink);
    color: #fff;
    font-size: 14px;
    font-weight: 600;
    text-decoration: none;
    border-radius: 0 0 4px 0;
  }

  .skip-nav:focus { left: 0; }

  .app {
    display: grid;
    grid-template-rows: auto 1fr auto;
    height: 100dvh;
    overflow: hidden;
    background: var(--bg-primary);
  }

  .main-content {
    display: flex;
    gap: 0;
    align-items: stretch;
    overflow: hidden;
    min-height: 0;
    border-bottom: 1px solid var(--border);
  }

  .thumbnail-panel {
    flex: 0 0 auto;
    border-right: 1px solid var(--border);
    padding: 8px;
    overflow: hidden;
  }

  .section-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }

  .thumb-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
  }

  .player-panel {
    flex: 0 0 auto;
    width: 315px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    justify-content: stretch;
    gap: 4px;
    border-right: 1px solid var(--border);
  }

  .vs-separator {
    text-align: center;
    color: var(--text-muted);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
  }

  .game-panel {
    flex: 1 1 auto;
    padding: 8px;
    overflow: hidden;
  }

  .game-view {
    display: flex;
    align-items: stretch;
    gap: 16px;
  }

  .board-area {
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    justify-content: center;
  }

  .eval-area {
    display: flex;
    flex-shrink: 0;
  }

  .info-area {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 40ch;
    overflow: hidden;
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

  .info-row .label { color: var(--text-secondary); }
  .info-row .value { color: var(--text-primary); }

  .result.in-progress { color: var(--accent-gold); }
  .result.terminal { color: var(--accent-teal); }

  .no-game {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    color: var(--text-muted);
    gap: 8px;
  }

  .no-game-hint { font-size: 12px; color: var(--text-muted); }

  .legend-area {
    flex: 1 1 auto;
    min-width: 0;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }

  .metrics-panel { padding: 12px 16px; }

  @media (max-width: 768px) {
    .main-content { flex-direction: column; }

    .thumbnail-panel {
      width: 100%;
      border-right: none;
      border-bottom: 1px solid var(--border);
      max-height: 160px;
    }

    .thumb-grid {
      grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    }

    .player-panel {
      width: 100%;
      flex-direction: row;
      border-right: none;
      border-bottom: 1px solid var(--border);
      justify-content: center;
    }

    .vs-separator { writing-mode: horizontal-tb; }

    .game-view { flex-direction: column; }
    .board-area { align-self: center; }
    .info-area { min-width: unset; }
  }

  @media (max-width: 480px) {
    .game-panel { padding: 8px; }
    .metrics-panel { padding: 8px; }
  }
</style>
