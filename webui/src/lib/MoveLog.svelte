<script>
  import { afterUpdate } from 'svelte'
  import { parseMoves, buildMoveRows } from './moveRows.js'

  export let moveHistoryJson = '[]'
  export let currentPlayer = 'black'

  let scrollContainer
  let notationStyle = 'western'
  const STYLES = ['western', 'japanese', 'usi']
  const STYLE_LABELS = { western: 'W', japanese: '漢', usi: 'USI' }
  const STYLE_NAMES = { western: 'Western', japanese: 'Japanese', usi: 'USI' }

  $: nextStyle = STYLES[(STYLES.indexOf(notationStyle) + 1) % STYLES.length]

  function toggleNotation() {
    notationStyle = nextStyle
  }

  $: moves = parseMoves(moveHistoryJson)
  $: rows = buildMoveRows(moves, notationStyle)

  afterUpdate(() => {
    if (scrollContainer) {
      scrollContainer.scrollTop = 0
    }
  })
</script>

<div class="move-log">
  <div class="header-row">
    <h2 class="header">Move Log</h2>
    <button class="notation-toggle" on:click={toggleNotation} title="Switch to {STYLE_NAMES[nextStyle]} notation" aria-label="Notation: {STYLE_NAMES[notationStyle]}. Click to switch to {STYLE_NAMES[nextStyle]}.">
      {STYLE_LABELS[notationStyle]}
    </button>
  </div>
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
          <tr class:latest={row.isLatest}>
            <td class="num">{row.num}</td>
            <td>{row.black}</td>
            <td>{row.white}</td>
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
  }

  .notation-toggle {
    background: none;
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 4px 10px;
    min-width: 36px;
    min-height: 36px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-muted);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .notation-toggle:hover {
    color: var(--text-primary);
    border-color: var(--text-secondary);
  }

  .notation-toggle:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }

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

  .empty {
    color: var(--text-muted);
    text-align: center;
    padding: 12px;
  }
</style>
