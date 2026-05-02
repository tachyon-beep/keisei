<script>
  import { showcaseGame, showcaseMoves, showcaseCurrentMove, sidecarAlive } from '../stores/showcase.js'
  import { safeParse } from './safeParse.js'
  import Board from './Board.svelte'
  import PieceTray from './PieceTray.svelte'
  import MoveLog from './MoveLog.svelte'
  import EvalBar from './EvalBar.svelte'
  import MatchControls from './MatchControls.svelte'
  import CommentaryPanel from './CommentaryPanel.svelte'
  import WinProbGraph from './WinProbGraph.svelte'
  import MatchQueue from './MatchQueue.svelte'

  $: move = $showcaseCurrentMove
  $: board = move ? safeParse(move.board_json, []) : []
  $: hands = move ? safeParse(move.hands_json, {}) : {}
  $: game = $showcaseGame
  $: moveHistoryJson = JSON.stringify(
    ($showcaseMoves || []).map(m => ({
      action: m.action_index,
      notation: m.usi_notation,
    }))
  )
</script>

<div id="showcase-main" class="showcase-view" tabindex="-1" aria-labelledby="tab-showcase">
  <MatchControls />
  {#if !$sidecarAlive}
    <div class="offline-banner">Showcase engine is offline. Start the sidecar to enable live matches.</div>
  {/if}
  {#if game}
    <div class="game-area">
      <div class="game-header">
        <span class="player black">{game.name_black} ({game.elo_black?.toFixed(0) ?? '?'})</span>
        <span class="vs">vs</span>
        <span class="player white">{game.name_white} ({game.elo_white?.toFixed(0) ?? '?'})</span>
        <span class="ply">Ply {game.total_ply}</span>
        {#if game.status !== 'in_progress'}
          <span class="result">{game.status.replaceAll('_', ' ')}</span>
        {/if}
      </div>
      <div class="game-content">
        <div class="board-side">
          <PieceTray color="white" hand={hands.white || {}} />
          <Board board={board} inCheck={!!move?.in_check} currentPlayer={move?.current_player || 'black'} />
          <PieceTray color="black" hand={hands.black || {}} />
        </div>
        <div class="eval-side">
          <EvalBar value={(move?.value_estimate ?? 0.5) * 2 - 1} currentPlayer={move?.current_player || 'black'} />
        </div>
        <div class="commentary-side">
          <CommentaryPanel />
          <WinProbGraph />
        </div>
        <div class="moves-side">
          <MoveLog moveHistoryJson={moveHistoryJson} currentPlayer={move?.current_player || 'black'} />
        </div>
      </div>
    </div>
  {:else}
    <div class="no-game">
      <p>No match in progress.</p>
      <p class="hint">Select two entries above and start a match!</p>
    </div>
  {/if}
  <MatchQueue />
</div>

<style>
  .showcase-view { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
  .showcase-view:focus { outline: none; }
  .offline-banner { padding: 8px 16px; background: var(--accent-gold); color: #000; font-size: 13px; font-weight: 600; text-align: center; }
  .game-area { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
  .game-header { display: flex; align-items: center; gap: 12px; padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 14px; }
  .player { font-weight: 600; }
  .player.black { color: var(--text-primary); }
  .player.white { color: var(--text-secondary); }
  .vs { color: var(--text-muted); font-size: 12px; }
  .ply { color: var(--text-muted); font-size: 12px; margin-left: auto; }
  .result { color: var(--accent-teal); font-weight: 600; text-transform: capitalize; }
  .game-content { flex: 1; display: flex; gap: 16px; padding: 8px; overflow: hidden; min-height: 0; }
  .board-side { display: flex; flex-direction: column; flex-shrink: 0; justify-content: center; }
  .eval-side { display: flex; flex-shrink: 0; }
  .commentary-side { display: flex; flex-direction: column; gap: 8px; width: 280px; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px; }
  .moves-side { flex: 1; min-width: 0; overflow: hidden; display: flex; flex-direction: column; }
  .no-game { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--text-muted); gap: 8px; }
  .hint { font-size: 13px; }
</style>
