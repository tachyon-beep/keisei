<script>
  import { tick } from 'svelte'
  import {
    showcaseGame, showcaseMoves, showcaseDisplayedMove, showcaseSelectedPly,
    isScrubbing, sidecarAlive, showcaseHeatmapEnabled,
  } from '../stores/showcase.js'
  import { safeParse } from './safeParse.js'
  import { parseUsi } from './usiCoords.js'
  import Board from './Board.svelte'
  import PieceTray from './PieceTray.svelte'
  import MoveLog from './MoveLog.svelte'
  import EvalBar from './EvalBar.svelte'
  import MatchControls from './MatchControls.svelte'
  import CommentaryPanel from './CommentaryPanel.svelte'
  import WinProbGraph from './WinProbGraph.svelte'
  import MatchQueue from './MatchQueue.svelte'
  import MatchScorecard from './MatchScorecard.svelte'
  import ShowcaseStatsBanner from './ShowcaseStatsBanner.svelte'

  // What the user is *looking at* (live tail OR scrubbed-to ply).
  $: move = $showcaseDisplayedMove
  $: board = move ? safeParse(move.board_json, []) : []
  $: hands = move ? safeParse(move.hands_json, {}) : {}
  $: game = $showcaseGame
  $: moves = $showcaseMoves
  $: scrubbing = $isScrubbing

  // The MoveLog needs all moves serialised; selectedIdx is its index into them.
  $: moveHistoryJson = JSON.stringify(
    (moves || []).map(m => ({
      action: m.action_index,
      notation: m.usi_notation,
    }))
  )
  $: selectedMoveIdx = (() => {
    if ($showcaseSelectedPly == null) return moves.length - 1
    return Math.max(0, Math.min($showcaseSelectedPly, moves.length - 1))
  })()

  // Last-move highlight for the displayed ply.
  $: lastMoveCoords = move?.move_usi ? parseUsi(move.move_usi) : null
  $: lastMoveFromIdx = lastMoveCoords?.fromIdx ?? -1
  $: lastMoveToIdx = lastMoveCoords?.toIdx ?? -1

  // Heatmap overlay — same logic as before but driven by displayed move.
  // Promotion variants (e.g., "5g5f" and "5g5f+") share a destination so we
  // sum probabilities into one combined "considered going to 5f" overlay.
  $: heatmap = (() => {
    if (!$showcaseHeatmapEnabled || !move?.move_heatmap_json) return null
    const raw = safeParse(move.move_heatmap_json, null)
    if (!raw || typeof raw !== 'object') return null
    const out = {}
    for (const [usi, prob] of Object.entries(raw)) {
      const parsed = parseUsi(usi)
      if (!parsed) continue
      out[parsed.toIdx] = (out[parsed.toIdx] ?? 0) + prob
    }
    return out
  })()

  // ── Scrubbing controls ─────────────────────────────────────────────
  function setSelected(idx) {
    if (idx == null || idx < 0) {
      // -1 / null both mean "return to live".
      showcaseSelectedPly.set(null)
      return
    }
    const clamped = Math.max(0, Math.min(idx, moves.length - 1))
    // If we land on the very last move, treat it as "live" so new moves
    // continue to advance the view automatically.
    if (clamped >= moves.length - 1) showcaseSelectedPly.set(null)
    else showcaseSelectedPly.set(clamped)
  }

  function step(delta) {
    if (moves.length === 0) return
    const current = $showcaseSelectedPly == null ? moves.length - 1 : $showcaseSelectedPly
    setSelected(current + delta)
  }

  function togglePin() {
    // Spacebar: pin to the currently-displayed ply, or release back to live.
    if ($showcaseSelectedPly == null) {
      // Currently live — pin to whatever ply we're showing.
      if (moves.length > 0) showcaseSelectedPly.set(moves.length - 1)
    } else {
      showcaseSelectedPly.set(null)
    }
  }

  function handleKeydown(e) {
    // Ignore when focus is in a form control or content-editable; spectators
    // should still be able to type into MatchControls' selects without us
    // hijacking arrow keys.
    const tag = (e.target?.tagName || '').toLowerCase()
    if (tag === 'input' || tag === 'select' || tag === 'textarea' || e.target?.isContentEditable) return
    if (e.metaKey || e.ctrlKey || e.altKey) return

    switch (e.key) {
      case 'ArrowLeft':
        e.preventDefault(); step(e.shiftKey ? -5 : -1); break
      case 'ArrowRight':
        e.preventDefault(); step(e.shiftKey ? 5 : 1); break
      case 'Home':
        e.preventDefault(); setSelected(0); break
      case 'End':
        e.preventDefault(); setSelected(-1); break  // -1 → live
      case ' ':
        e.preventDefault(); togglePin(); break
      case 'h':
      case 'H':
        e.preventDefault(); showcaseHeatmapEnabled.update(v => !v); break
    }
  }

  function onMoveSelect(event) {
    setSelected(event.detail.idx)
  }

  function onHeatmapToggle() {
    showcaseHeatmapEnabled.update(v => !v)
  }

  // ── Move announcer (aria-live) ─────────────────────────────────────
  // Polite announcer that emits a plain-English description of each new move
  // arriving on the wire. We watch only the currentMove (live tail), not the
  // displayed/scrubbed move — otherwise scrubbing would spam the screen reader.
  let announcement = ''
  let lastAnnouncedPly = null

  $: if (moves.length > 0) {
    const live = moves[moves.length - 1]
    if (live && live.ply !== lastAnnouncedPly) {
      lastAnnouncedPly = live.ply
      const side = live.current_player === 'black' ? 'White' : 'Black'  // who just moved
      const usi = live.usi_notation || live.move_usi || '(unknown)'
      announcement = `Ply ${live.ply}: ${side} plays ${usi}`
    }
  } else if (moves.length === 0 && lastAnnouncedPly != null) {
    lastAnnouncedPly = null
    announcement = ''
  }
</script>

<svelte:window on:keydown={handleKeydown} />

<div id="showcase-main" class="showcase-view" tabindex="-1" aria-labelledby="tab-showcase">
  <!-- Polite live region for screen readers — announces new live moves only. -->
  <div class="sr-only" role="status" aria-live="polite" aria-atomic="true">{announcement}</div>

  <ShowcaseStatsBanner />

  {#if !$sidecarAlive}
    <div class="offline-banner" role="alert">
      <strong>Showcase engine is offline.</strong> Start the sidecar to enable live matches.
    </div>
  {/if}

  <!-- Setup: collapsed when a match is in progress so the board is the hero. -->
  <MatchControls collapsed={game != null} />

  {#if game}
    <MatchScorecard {game} displayedMove={move} {scrubbing} />

    <div class="game-content">
      <div class="board-side">
        <PieceTray color="white" hand={hands.white || {}} />
        <Board
          board={board}
          inCheck={!!move?.in_check}
          currentPlayer={move?.current_player || 'black'}
          lastMoveFromIdx={lastMoveFromIdx}
          lastMoveToIdx={lastMoveToIdx}
          heatmap={heatmap}
        />
        <PieceTray color="black" hand={hands.black || {}} />
        <div class="analysis-cluster" role="group" aria-label="Board analysis overlays">
          <button
            class="heatmap-toggle"
            on:click={onHeatmapToggle}
            aria-pressed={$showcaseHeatmapEnabled}
            title="Toggle policy heatmap overlay (h)"
          >
            Heatmap: {$showcaseHeatmapEnabled ? 'On' : 'Off'}
          </button>
          {#if $showcaseHeatmapEnabled}
            <div class="heatmap-legend" aria-label="Heatmap probability scale">
              <span class="legend-label">low</span>
              <span class="legend-ramp" aria-hidden="true"></span>
              <span class="legend-label">high</span>
            </div>
          {/if}
        </div>
      </div>

      <div class="eval-side">
        <EvalBar value={(move?.value_estimate ?? 0.5) * 2 - 1} currentPlayer={move?.current_player || 'black'} />
      </div>

      <div class="commentary-side">
        <CommentaryPanel />
        <WinProbGraph />
        <div class="kbd-hints" aria-label="Keyboard shortcuts">
          <span class="kbd">←</span><span class="kbd">→</span> step
          · <span class="kbd">Shift</span>+arrows ×5
          · <span class="kbd">Home</span>/<span class="kbd">End</span> first/live
          · <span class="kbd">Space</span> pin
          · <span class="kbd">H</span> heatmap
        </div>
      </div>

      <div class="moves-side">
        <MoveLog
          moveHistoryJson={moveHistoryJson}
          interactive={true}
          selectedIdx={selectedMoveIdx}
          on:select={onMoveSelect}
        />
      </div>
    </div>
  {:else}
    <div class="no-game">
      <div class="no-game-card">
        <h2>No match in progress</h2>
        <p class="hint">Showcase lets you watch any two players from the league face off live.</p>
        <ol class="howto">
          <li>Pick a Black and a White entry above.</li>
          <li>Choose a playback speed.</li>
          <li>Press <strong>Start Match</strong>.</li>
        </ol>
        <p class="hint subtle">
          Once a match is running you can scrub history with ←/→, pause with Space,
          and toggle the policy heatmap with H.
        </p>
      </div>
    </div>
  {/if}

  <MatchQueue />
</div>

<style>
  .showcase-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }
  .showcase-view:focus { outline: none; }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  .offline-banner {
    padding: 8px 16px;
    background: var(--badge-bg-danger);
    color: var(--danger);
    border-bottom: 1px solid var(--danger);
    font-size: 13px;
    text-align: center;
  }

  .game-content {
    flex: 1;
    display: flex;
    gap: 16px;
    padding: 8px;
    overflow: hidden;
    min-height: 0;
  }

  .board-side {
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    justify-content: center;
    gap: 6px;
  }

  .analysis-cluster {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 8px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-secondary);
    align-self: stretch;
    justify-content: space-between;
    flex-wrap: wrap;
  }

  .heatmap-toggle {
    padding: 4px 10px;
    min-height: 32px;
    background: var(--bg-primary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }
  .heatmap-toggle[aria-pressed='true'] {
    background: var(--accent-teal);
    color: var(--bg-primary);
    border-color: var(--accent-teal);
  }
  .heatmap-toggle:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }

  .heatmap-legend {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .legend-ramp {
    display: inline-block;
    width: 60px;
    height: 8px;
    border-radius: 2px;
    /* Match the heatmap overlay's opacity ramp from Board.svelte. */
    background: linear-gradient(
      to right,
      rgba(77, 184, 168, 0.15),
      rgba(77, 184, 168, 0.65)
    );
  }

  .eval-side { display: flex; flex-shrink: 0; }

  .commentary-side {
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 300px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
  }

  .kbd-hints {
    padding: 6px 8px;
    font-size: 11px;
    color: var(--text-muted);
    border-top: 1px solid var(--border-subtle);
    line-height: 1.6;
  }
  .kbd {
    display: inline-block;
    padding: 0 5px;
    margin: 0 1px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-family: monospace;
    font-size: 10px;
    color: var(--text-secondary);
  }

  .moves-side {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .no-game {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }
  .no-game-card {
    max-width: 460px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px 28px;
    color: var(--text-primary);
  }
  .no-game-card h2 {
    margin: 0 0 8px;
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
  }
  .hint { font-size: 13px; color: var(--text-secondary); margin: 0 0 12px; }
  .hint.subtle { font-size: 12px; color: var(--text-muted); margin-top: 12px; }
  .howto {
    margin: 0 0 8px;
    padding-left: 20px;
    color: var(--text-secondary);
    font-size: 13px;
  }
  .howto li { margin-bottom: 4px; }

  @media (max-width: 768px) {
    .game-content {
      flex-direction: column;
      gap: 8px;
      overflow-y: auto;
    }
    .board-side { align-self: center; }
    .analysis-cluster { align-self: stretch; }
    .eval-side { display: none; }  /* Hidden on mobile — eval shown in Commentary panel */
    .commentary-side {
      width: 100%;
      max-height: 220px;
    }
    .moves-side {
      max-height: 260px;
    }
  }
</style>
