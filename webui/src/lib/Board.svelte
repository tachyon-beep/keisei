<script>
  import { pieceKanji } from './pieces.js'

  export let board = []
  export let inCheck = false
  export let currentPlayer = 'black'
  /** Index (0-80) of the last move's destination square, or -1. */
  export let lastMoveIdx = -1

  const colLabels = [9, 8, 7, 6, 5, 4, 3, 2, 1]
  const rowLabels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
</script>

<div class="board-container" aria-label="9×9 shogi board, {currentPlayer} to move{inCheck ? ', in check' : ''}">
  <div class="col-labels" aria-hidden="true">
    {#each colLabels as label}
      <span>{label}</span>
    {/each}
  </div>

  <div class="board-with-rows">
    <div class="board" role="grid" aria-label="Board squares">
      {#each Array(81) as _, idx}
        {@const piece = board[idx]}
        {@const col = colLabels[idx % 9]}
        {@const row = rowLabels[Math.floor(idx / 9)]}
        <div
          class="square"
          role="gridcell"
          aria-label="{col}{row}{piece ? ': ' + (piece.color === 'white' ? 'White ' : 'Black ') + piece.type + (piece.promoted ? ' (promoted)' : '') : ''}"
          class:has-piece={piece != null}
          class:last-move={idx === lastMoveIdx}
        >
          {#if piece}
            <span
              class="piece"
              class:white={piece.color === 'white'}
              class:promoted={piece.promoted}
              lang="ja"
              title="{piece.color} {piece.type}{piece.promoted ? ' (promoted)' : ''}"
            >
              {pieceKanji(piece.type, piece.promoted, piece.color)}
            </span>
          {/if}
        </div>
      {/each}
    </div>

    <div class="row-labels" aria-hidden="true">
      {#each rowLabels as label}
        <span>{label}</span>
      {/each}
    </div>
  </div>
</div>

<style>
  .board-container {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .col-labels {
    display: grid;
    grid-template-columns: repeat(9, 1fr);
    padding-left: 0;
    padding-right: 20px;
    font-size: 11px;
    color: var(--text-muted);
    text-align: center;
  }

  .board-with-rows {
    display: flex;
    gap: 4px;
  }

  .board {
    display: grid;
    grid-template-columns: repeat(9, 36px);
    grid-template-rows: repeat(9, 36px);
    border: 2px solid var(--border-board);
    background: var(--bg-board);
  }

  .square {
    border: 1px solid var(--border-board-inner);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    position: relative;
  }

  .square.last-move {
    background: rgba(74, 222, 128, 0.25);
    border-color: var(--accent-green);
  }

  .piece {
    cursor: default;
    user-select: none;
    line-height: 1;
  }

  .piece.white {
    transform: rotate(180deg);
  }

  .piece.promoted {
    color: var(--promoted);
  }

  .row-labels {
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    font-size: 11px;
    color: var(--text-muted);
    padding: 0 2px;
  }
</style>
