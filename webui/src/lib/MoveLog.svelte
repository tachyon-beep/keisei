<script>
  import { afterUpdate, createEventDispatcher } from 'svelte'
  import { parseMoves, buildMoveRows } from './moveRows.js'
  import { notationStyle } from '../stores/notation.js'
  import NotationToggle from './NotationToggle.svelte'

  export let moveHistoryJson = '[]'
  /**
   * Index (0-based) of the currently displayed move in the moves array, or -1
   * for "live tail" / no selection. When >= 0 the matching cell is highlighted
   * and an "← Live" affordance is shown so users can return to follow-mode.
   */
  export let selectedIdx = -1
  /**
   * When true, move cells become clickable and emit a 'select' event with the
   * 0-based move index. Defaults to false so existing read-only callers (the
   * Training tab) keep their current behavior.
   */
  export let interactive = false

  const dispatch = createEventDispatcher()

  let scrollContainer

  $: moves = parseMoves(moveHistoryJson)
  $: rows = buildMoveRows(moves, $notationStyle)
  $: totalMoves = moves.length
  $: scrubbing = interactive && selectedIdx >= 0 && selectedIdx < totalMoves - 1

  function selectIdx(idx) {
    if (!interactive) return
    if (idx < 0 || idx >= totalMoves) return
    dispatch('select', { idx })
  }

  function returnToLive() {
    dispatch('select', { idx: -1 })
  }

  function handleCellKeydown(e, idx) {
    if (!interactive) return
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      selectIdx(idx)
    }
  }

  afterUpdate(() => {
    if (scrollContainer && !scrubbing) {
      scrollContainer.scrollTop = 0
    }
  })
</script>

<div class="move-log">
  <div class="header-row">
    <h2 class="header">Move Log</h2>
    {#if scrubbing}
      <button
        class="live-btn"
        on:click={returnToLive}
        title="Return to the live position (End)"
        aria-label="Return to live position"
      >
        ← Live
      </button>
    {/if}
    <NotationToggle />
  </div>
  <!-- svelte-ignore a11y-no-noninteractive-tabindex -->
  <!-- Scrollable region: keyboard users need to focus this to scroll the move history with arrow keys. -->
  <div class="table-container" role="log" aria-label="Move history" tabindex="0" bind:this={scrollContainer}>
    <table>
      <thead>
        <tr>
          <th class="num">#</th>
          <th>☗ Black</th>
          <th>☖ White</th>
        </tr>
      </thead>
      <tbody>
        {#each [...rows].reverse() as row}
          {@const blackIdx = (row.num - 1) * 2}
          {@const whiteIdx = blackIdx + 1}
          {@const blackHasMove = blackIdx < totalMoves}
          {@const whiteHasMove = whiteIdx < totalMoves}
          <tr class:latest={row.isLatest && !scrubbing}>
            <td class="num">{row.num}</td>
            <td
              class:cell={interactive && blackHasMove}
              class:selected={interactive && blackIdx === selectedIdx}
              on:click={() => blackHasMove && selectIdx(blackIdx)}
              on:keydown={(e) => handleCellKeydown(e, blackIdx)}
              role={interactive && blackHasMove ? 'button' : undefined}
              tabindex={interactive && blackHasMove ? 0 : undefined}
              aria-pressed={interactive && blackHasMove ? blackIdx === selectedIdx : undefined}
            >{row.black}</td>
            <td
              class:cell={interactive && whiteHasMove}
              class:selected={interactive && whiteIdx === selectedIdx}
              on:click={() => whiteHasMove && selectIdx(whiteIdx)}
              on:keydown={(e) => handleCellKeydown(e, whiteIdx)}
              role={interactive && whiteHasMove ? 'button' : undefined}
              tabindex={interactive && whiteHasMove ? 0 : undefined}
              aria-pressed={interactive && whiteHasMove ? whiteIdx === selectedIdx : undefined}
            >{row.white}</td>
          </tr>
        {/each}
        {#if rows.length === 0}
          <tr><td colspan="3" class="empty">No moves yet</td></tr>
        {/if}
      </tbody>
    </table>
  </div>
</div>

<style>
  .move-log {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    flex: 1;
    min-height: 0;
  }

  .header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 6px;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border-subtle);
    position: sticky;
    top: 0;
    background: var(--bg-primary);
  }

  h2.header {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    flex: 1;
  }

  .live-btn {
    background: var(--badge-bg-gold);
    border: 1px solid var(--accent-gold);
    border-radius: 3px;
    color: var(--accent-gold);
    cursor: pointer;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 4px 8px;
    text-transform: uppercase;
  }

  .live-btn:hover { background: var(--accent-gold); color: var(--bg-primary); }
  .live-btn:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }

  .table-container {
    overflow-y: scroll;
    flex: 1;
    min-height: 0;
    padding: 4px;
  }

  .table-container:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: -2px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-family: monospace;
    font-size: 14px;
  }

  thead tr {
    color: var(--text-muted);
    font-size: 13px;
  }

  th, td {
    text-align: left;
    padding: 2px 6px;
  }

  th.num, td.num {
    width: 30px;
    color: var(--text-muted);
  }

  tbody tr {
    color: var(--text-primary);
  }

  tbody tr:nth-child(even) {
    background: var(--bg-secondary);
  }

  tr.latest {
    background: var(--bg-selected) !important;
    border-left: 2px solid var(--accent-teal);
  }

  tr.latest td {
    color: var(--accent-teal);
  }

  td.cell {
    cursor: pointer;
    border-radius: 2px;
  }

  td.cell:hover {
    background: var(--bg-selected);
    color: var(--accent-teal);
  }

  td.cell:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: -2px;
  }

  td.selected {
    background: var(--bg-selected) !important;
    color: var(--accent-gold) !important;
    font-weight: 700;
    box-shadow: inset 2px 0 0 var(--accent-gold);
  }

  .empty {
    color: var(--text-muted);
    text-align: center;
    padding: 12px;
  }
</style>
