# Plan 5: Game Thumbnails + Move Log

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the game thumbnail grid (click to focus) and the move log (two-column Black/White table with auto-scroll).

**Architecture:** Thumbnails are mini Board components (scaled down). Clicking a thumbnail updates `selectedGameId` store. Move log parses `move_history_json` and renders as a two-column table.

**Tech Stack:** Svelte, CSS

---

### Task 1: GameThumbnail Component

**Files:**
- Create: `webui/src/lib/GameThumbnail.svelte`

- [ ] **Step 1: Implement GameThumbnail.svelte**

`webui/src/lib/GameThumbnail.svelte`:
```svelte
<script>
  import { selectedGameId } from '../stores/games.js'
  import { pieceKanji } from './pieces.js'

  /** @type {Object} game snapshot */
  export let game

  $: selected = $selectedGameId === game.game_id
  $: board = typeof game.board_json === 'string' ? JSON.parse(game.board_json) : (game.board || [])
  $: statusText = game.is_over
    ? game.result.replace('_', ' ')
    : `Ply ${game.ply}`

  function handleClick() {
    selectedGameId.set(game.game_id)
  }
</script>

<button class="thumbnail" class:selected on:click={handleClick}>
  <div class="mini-board">
    {#each Array(81) as _, idx}
      {@const piece = board[idx]}
      <div class="cell">
        {#if piece}
          <span class:white={piece.color === 'white'} class:promoted={piece.promoted}>
            {pieceKanji(piece.type, piece.promoted, piece.color)}
          </span>
        {/if}
      </div>
    {/each}
  </div>
  <div class="label">
    G{game.game_id + 1} — {statusText}
  </div>
</button>

<style>
  .thumbnail {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px;
    cursor: pointer;
    text-align: center;
    transition: border-color 0.15s;
    width: 100%;
  }

  .thumbnail:hover {
    border-color: var(--text-secondary);
  }

  .thumbnail.selected {
    border: 2px solid var(--accent-green);
    background: #1a3a2a;
  }

  .mini-board {
    display: grid;
    grid-template-columns: repeat(9, 1fr);
    grid-template-rows: repeat(9, 1fr);
    width: 72px;
    height: 72px;
    margin: 0 auto;
    background: var(--bg-board);
    border-radius: 2px;
  }

  .cell {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 6px;
    line-height: 1;
  }

  .cell span.white {
    transform: rotate(180deg);
  }

  .cell span.promoted {
    color: #c00;
  }

  .label {
    font-size: 10px;
    color: var(--text-secondary);
    margin-top: 4px;
  }

  .thumbnail.selected .label {
    color: var(--accent-green);
  }
</style>
```

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/GameThumbnail.svelte
git commit -m "feat: clickable game thumbnail with mini board preview"
```

---

### Task 2: MoveLog Component

**Files:**
- Create: `webui/src/lib/MoveLog.svelte`

- [ ] **Step 1: Implement MoveLog.svelte**

`webui/src/lib/MoveLog.svelte`:
```svelte
<script>
  import { afterUpdate } from 'svelte'

  /** @type {string} JSON string of move history array */
  export let moveHistoryJson = '[]'

  /** @type {string} */
  export let currentPlayer = 'black'

  let scrollContainer

  $: moves = typeof moveHistoryJson === 'string'
    ? JSON.parse(moveHistoryJson)
    : moveHistoryJson

  // Pair moves into rows: [moveNum, blackMove, whiteMove]
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
  <div class="header">Move Log</div>
  <div class="table-container" bind:this={scrollContainer}>
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
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .header {
    font-size: 10px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 8px 10px;
    border-bottom: 1px solid #222;
    position: sticky;
    top: 0;
    background: #0d1117;
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
    background: #1a3a2a !important;
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
```

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/MoveLog.svelte
git commit -m "feat: move log with two-column layout and auto-scroll"
```
