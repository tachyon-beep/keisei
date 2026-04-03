<script>
  import { afterUpdate } from 'svelte'
  import { parseMoves, buildMoveRows } from './moveRows.js'

  export let moveHistoryJson = '[]'
  export let currentPlayer = 'black'

  let scrollContainer

  $: moves = parseMoves(moveHistoryJson)
  $: rows = buildMoveRows(moves)

  afterUpdate(() => {
    if (scrollContainer) {
      scrollContainer.scrollTop = scrollContainer.scrollHeight
    }
  })
</script>

<div class="move-log">
  <h2 class="header">Move Log</h2>
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
        {#each rows as row}
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

  h2.header {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border-subtle);
    position: sticky;
    top: 0;
    background: var(--bg-primary);
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
