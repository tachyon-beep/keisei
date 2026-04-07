<script>
  import { pieceKanji } from './pieces.js'

  export let board = []
  export let inCheck = false
  export let currentPlayer = 'black'
  /** Index (0-80) of the last move's destination square, or -1. */
  export let lastMoveIdx = -1

  const colLabels = [9, 8, 7, 6, 5, 4, 3, 2, 1]
  const rowLabels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

  // Summarize piece positions for screen readers (view-only board)
  $: boardDescription = (() => {
    const counts = { black: 0, white: 0 }
    for (const piece of board) {
      if (piece) counts[piece.color]++
    }
    return `Black has ${counts.black} pieces, White has ${counts.white} pieces on the board`
  })()
</script>

<div class="board-container" role="img" aria-label="9×9 shogi board, {currentPlayer} to move{inCheck ? ', in check' : ''}. {boardDescription}">
  <div class="col-labels" aria-hidden="true">
    {#each colLabels as label}
      <span>{label}</span>
    {/each}
  </div>

  <div class="board-with-rows">
    <div class="board" aria-hidden="true">
      {#each Array(81) as _, idx}
        {@const piece = board[idx]}
        <div
          class="square"
          class:has-piece={piece != null}
          class:last-move={idx === lastMoveIdx}
        >
          {#if piece}
            <span
              class="piece"
              class:white={piece.color === 'white'}
              class:promoted={piece.promoted}
              lang="ja"
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
    font-size: 12px;
    color: var(--text-secondary);
    text-align: center;
    background: var(--bg-board-label);
    border-radius: 4px 4px 0 0;
    padding-top: 2px;
    padding-bottom: 2px;
  }

  .board-with-rows {
    display: flex;
    gap: 4px;
  }

  .board {
    --sq: min(72px, (100vw - 60px) / 9);
    display: grid;
    grid-template-columns: repeat(9, var(--sq));
    grid-template-rows: repeat(9, var(--sq));
    border: 2px solid var(--border-board);
    background: var(--bg-board);
  }

  .square {
    border: 1px solid var(--border-board-inner);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: min(36px, calc(var(--sq) * 0.5));
    position: relative;
  }

  .square.last-move {
    background: var(--bg-last-move);
    border-color: var(--accent-teal);
  }

  .piece {
    cursor: default;
    user-select: none;
    line-height: 1;
    color: var(--text-piece);
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
    font-size: 12px;
    color: var(--text-secondary);
    padding: 2px 4px;
    background: var(--bg-board-label);
    border-radius: 0 4px 4px 0;
  }
</style>
