<script>
  import { afterUpdate } from 'svelte'

  export let moveHistoryJson = '[]'
  export let currentPlayer = 'black'

  let scrollContainer

  $: moves = (() => {
    try { return typeof moveHistoryJson === 'string' ? JSON.parse(moveHistoryJson) : (moveHistoryJson || []) }
    catch { return [] }
  })()

  $: rows = (() => {
    const result = []
    for (let i = 0; i < moves.length; i += 2) {
      result.push({
        num: Math.floor(i / 2) + 1,
        black: moves[i]?.notation || '',
        white: moves[i + 1]?.notation || '',
        isLatest: i >= moves.length - 2,
      })
    }
    return result
  })()

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
    overflow-y: auto;
    max-height: 300px;
    padding: 4px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-family: monospace;
    font-size: 12px;
  }

  thead tr {
    color: var(--text-muted);
    font-size: 10px;
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
    border-left: 2px solid var(--accent-green);
  }

  tr.latest td {
    color: var(--accent-green);
  }

  .empty {
    color: var(--text-muted);
    text-align: center;
    padding: 12px;
  }
</style>
