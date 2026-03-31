<script>
  import { pieceKanji } from './pieces.js'

  export let board = []
  export let inCheck = false
  export let currentPlayer = 'black'

  const colLabels = [9, 8, 7, 6, 5, 4, 3, 2, 1]
  const rowLabels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
</script>

<div class="board-container">
  <div class="col-labels">
    {#each colLabels as label}
      <span>{label}</span>
    {/each}
  </div>

  <div class="board-with-rows">
    <div class="board">
      {#each Array(81) as _, idx}
        {@const piece = board[idx]}
        {@const row = Math.floor(idx / 9)}
        {@const col = idx % 9}
        <div
          class="square"
          class:has-piece={piece != null}
        >
          {#if piece}
            <span
              class="piece"
              class:white={piece.color === 'white'}
              class:promoted={piece.promoted}
            >
              {pieceKanji(piece.type, piece.promoted, piece.color)}
            </span>
          {/if}
        </div>
      {/each}
    </div>

    <div class="row-labels">
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
    border: 2px solid #8B7355;
    background: var(--bg-board);
  }

  .square {
    border: 1px solid #b8956a;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    position: relative;
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
    color: #c00;
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
